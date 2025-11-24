import os.path as osp
import glob
import logging
import insightface
import cv2
import numpy as np
from insightface.model_zoo.model_zoo import ModelRouter, PickableInferenceSession, get_default_providers
from insightface.model_zoo.retinaface import RetinaFace
from insightface.model_zoo.landmark import Landmark
from insightface.model_zoo.attribute import Attribute
from insightface.model_zoo.inswapper import INSwapper
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.app import FaceAnalysis
from insightface.utils import DEFAULT_MP_NAME, ensure_available, face_align
from insightface.model_zoo import model_zoo
import onnxruntime
import onnx
from onnx import numpy_helper
from scripts.reactor_logger import logger


def _get_input_shape_dims(input_info):
    try:
        return len(input_info.shape)
    except Exception:
        return 0


def _find_blob_and_latent_inputs(inputs):
    blob_input = None
    latent_input = None
    for inp in inputs:
        dims = _get_input_shape_dims(inp)
        if dims == 4:
            blob_input = inp
        elif dims == 2:
            latent_input = inp
    return blob_input, latent_input


def patched_get_model(self, **kwargs):
    session = PickableInferenceSession(self.onnx_file, **kwargs)
    inputs = session.get_inputs()
    input_cfg = inputs[0]
    input_shape = input_cfg.shape
    outputs = session.get_outputs()

    if len(outputs) >= 5:
        return RetinaFace(model_file=self.onnx_file, session=session)

    dims = _get_input_shape_dims(input_cfg)
    if dims >= 4 and input_shape[2] == 192 and input_shape[3] == 192:
        return Landmark(model_file=self.onnx_file, session=session)
    elif dims >= 4 and input_shape[2] == 96 and input_shape[3] == 96:
        return Attribute(model_file=self.onnx_file, session=session)

    if len(inputs) == 2:
        blob_input, latent_input = _find_blob_and_latent_inputs(inputs)
        if blob_input is not None and latent_input is not None:
            blob_shape = blob_input.shape
            if blob_shape[2] == 128 and blob_shape[3] == 128:
                return INSwapper(model_file=self.onnx_file, session=session)
            if blob_shape[2] == blob_shape[3] and blob_shape[2] >= 192:
                return Hyperswapper(model_file=self.onnx_file, session=session)

    if dims >= 4 and input_shape[2] == input_shape[3] and input_shape[2] >= 112 and input_shape[2] % 16 == 0:
        return ArcFaceONNX(model_file=self.onnx_file, session=session)
    return None


def patched_faceanalysis_init(self, name=DEFAULT_MP_NAME, root='~/.insightface', allowed_modules=None, **kwargs):
    onnxruntime.set_default_logger_severity(3)
    self.models = {}
    self.model_dir = ensure_available('models', name, root=root)
    onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
    onnx_files = sorted(onnx_files)
    for onnx_file in onnx_files:
        model = model_zoo.get_model(onnx_file, **kwargs)
        if model is None:
            print('model not recognized:', onnx_file)
        elif allowed_modules is not None and model.taskname not in allowed_modules:
            print('model ignore:', onnx_file, model.taskname)
            del model
        elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
            self.models[model.taskname] = model
        else:
            print('duplicated model task type, ignore:', onnx_file, model.taskname)
            del model
    assert 'detection' in self.models
    self.det_model = self.models['detection']


def patched_faceanalysis_prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
    self.det_thresh = det_thresh
    assert det_size is not None
    self.det_size = det_size
    for taskname, model in self.models.items():
        if taskname == 'detection':
            model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
        else:
            model.prepare(ctx_id)


def _latent_dim_from_initializers(graph):
    """Try to infer the latent embedding size from ONNX initializers.

    The goal is to combine the largest plausible projection matrix with
    any explicit latent-sized tensor we see, while skipping tiny kernels
    that previously caused conflicts when merging with upstream changes.
    """

    preferred = [512, 320, 256, 192, 160, 128, 96, 80, 64]
    best = None

    # Track all candidate dimensions we see to reconcile conflicting edits.
    seen_dims = set()

    for initializer in graph.initializer:
        dims = tuple(initializer.dims)
        if len(dims) != 2:
            continue
        if dims[0] == dims[1]:
            seen_dims.add(dims[0])
        seen_dims.update(dims)

    for candidate in preferred:
        if candidate in seen_dims:
            best = candidate
            break

    if best is None and seen_dims:
        # Fall back to the largest square-ish matrix we saw to remain
        # compatible with upstream logic while still preferring projection
        # shapes over small conv kernels.
        fallback_dims = [d for d in seen_dims if d is not None and d >= 64]
        if fallback_dims:
            best = max(fallback_dims)

    return best


def _reshape_emap(best, latent_dim):
    """Force the embedding map to emit vectors with the expected latent size.

    Some hyperswap variants store projection matrices that expand embeddings
    (e.g., 512x1024). That leads to ONNX Runtime complaining that the "source"
    input has the wrong dimension. We truncate or pad to a square latent_dim
    projection so INSwapper.get always returns the shape the model expects.
    """

    if latent_dim is None or best.ndim != 2:
        return np.asarray(best, dtype=np.float32)

    rows, cols = best.shape
    if rows != latent_dim or cols != latent_dim:
        logger.warning(
            "Reshaping embedding map from %sx%s to %sx%s to align with latent dim.",
            rows,
            cols,
            latent_dim,
            latent_dim,
        )

    # Align columns first so any subsequent row padding has the correct width.
    if cols < latent_dim:
        pad_cols = latent_dim - cols
        best = np.hstack([best, np.eye(rows, dtype=np.float32)[:, :pad_cols]])
    elif cols > latent_dim:
        best = best[:, :latent_dim]

    # Adjust rows to guarantee the dot product input matches latent_dim.
    if best.shape[0] < latent_dim:
        pad_rows = latent_dim - best.shape[0]
        best = np.vstack([
            best,
            np.eye(latent_dim, dtype=np.float32)[:pad_rows, :latent_dim],
        ])
    elif best.shape[0] > latent_dim:
        best = best[:latent_dim, :]

    return np.asarray(best, dtype=np.float32)


def _pick_emap(graph, latent_dim):
    """
    Hyperswap models may not store the embedding map as the last initializer
    or may expose a 1D bias vector there, which breaks the stock dot product
    against arcface embeddings. Choose a matrix whose shape aligns with the
    latent size (guessing common dimensions when metadata is missing),
    transposing if needed, and fall back to an identity map so inference can
    proceed even when the ONNX layout differs.
    """

    best = None
    if latent_dim is None or latent_dim < 64:
        if latent_dim is not None and latent_dim < 64:
            logger.warning(
                "Ignoring suspicious latent dim %s; probing ONNX graph for a realistic value instead.",
                latent_dim,
            )
        latent_dim = _latent_dim_from_initializers(graph) or 512

    allowed_dims = {64, 80, 96, 128, 160, 192, 256, 320, 512, latent_dim}
    min_dim = 64 if latent_dim is None else min(latent_dim, 64)

    for initializer in graph.initializer:
        arr = numpy_helper.to_array(initializer)
        if arr.ndim != 2:
            continue

        if min(arr.shape) < min_dim:
            continue

        if arr.shape[0] not in allowed_dims and arr.shape[1] not in allowed_dims:
            continue

        # Prefer matrices that already emit the latent size so we don't upsample
        # embeddings for models that expect 512-dim inputs.
        if arr.shape == (latent_dim, latent_dim):
            best = arr
            break
        if arr.shape[1] == latent_dim:
            best = arr
            break
        if arr.shape[0] == latent_dim:
            best = arr
            break

        # If neither axis matches exactly, prefer the larger square-ish option
        # to stay compatible with upstream heuristics while avoiding tiny kernels.
        if best is None and arr.shape[0] == arr.shape[1] and arr.shape[0] in allowed_dims:
            best = arr

    if best is None and latent_dim is not None:
        logger.warning(
            "No embedding map matched latent dim %s; using identity matrix so inference can continue.",
            latent_dim,
        )
        best = np.eye(latent_dim, dtype=np.float32)

    if best is None:
        logger.warning(
            "No embedding map found in ONNX graph; defaulting to %sx%s identity passthrough.",
            latent_dim,
            latent_dim,
        )
        best = np.eye(latent_dim, dtype=np.float32)

    if best.ndim == 1:
        best = np.diag(best.astype(np.float32))

    if latent_dim is not None:
        best = _reshape_emap(best, latent_dim)

    return np.asarray(best, dtype=np.float32)


def _infer_latent_dim(inputs, graph):
    latent_dim = None
    if len(inputs) > 1:
        latent_shape = inputs[1].shape
        if len(latent_shape) > 1 and latent_shape[1] is not None:
            latent_dim = latent_shape[1]
    if latent_dim is None and len(graph.input) > 1:
        latent_vi = graph.input[1]
        if latent_vi.type.HasField("tensor_type"):
            dims = []
            for dim in latent_vi.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    dims.append(dim.dim_value)
                elif dim.HasField("dim_param"):
                    try:
                        dims.append(int(dim.dim_param))
                    except (TypeError, ValueError):
                        dims.append(None)
                else:
                    dims.append(None)
            if len(dims) > 1 and dims[1] is not None:
                latent_dim = dims[1]

    if latent_dim is not None and latent_dim < 64:
        logger.warning(
            "Latent dim %s reported by ONNX metadata is too small for embeddings; falling back to graph inspection.",
            latent_dim,
        )
        return None

    return latent_dim


def patched_inswapper_init(self, model_file=None, session=None):
    self.model_file = model_file
    self.session = session
    model = onnx.load(self.model_file)
    graph = model.graph
    self.emap = numpy_helper.to_array(graph.initializer[-1])
    self.input_mean = 0.0
    self.input_std = 255.0
    if self.session is None:
        self.session = onnxruntime.InferenceSession(self.model_file, None)
    inputs = self.session.get_inputs()
    self.input_names = []
    for inp in inputs:
        self.input_names.append(inp.name)
    outputs = self.session.get_outputs()
    output_names = []
    for out in outputs:
        output_names.append(out.name)
    self.output_names = output_names
    assert len(self.output_names) == 1
    input_cfg = inputs[0]
    input_shape = input_cfg.shape
    self.input_shape = input_shape
    self.input_size = tuple(input_shape[2:4][::-1])


def _extract_embedding_map(graph, latent_dim):
    for initializer in graph.initializer:
        arr = numpy_helper.to_array(initializer)
        if arr.shape == (latent_dim, latent_dim):
            return arr
    return np.eye(latent_dim, dtype=np.float32)


class Hyperswapper():
    taskname = 'inswapper'

    def __init__(self, model_file=None, session=None):
        self.model_file = model_file
        self.session = session
        model = onnx.load(self.model_file)
        graph = model.graph
        inputs = graph.input
        outputs = graph.output
        blob_input, latent_input = _find_blob_and_latent_inputs(inputs)
        self.input_mean = 0.0
        self.input_std = 255.0
        self.blob_name = blob_input.name
        self.latent_name = latent_input.name
        self.latent_dim = latent_input.type.tensor_type.shape.dim[-1].dim_value
        self.emap = _extract_embedding_map(graph, self.latent_dim)
        self.output_names = [out.name for out in outputs]
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        self.input_size = tuple(blob_input.type.tensor_type.shape.dim[2:4][::-1])

    def _normalize_latent(self, embedding):
        latent = embedding.reshape((1, -1)).astype(np.float32)
        latent = np.dot(latent, self.emap)
        norm = np.linalg.norm(latent)
        if norm != 0:
            latent /= norm
        return latent

    def _prepare_inputs(self, img, target_face, source_face):
        aimg, M = face_align.norm_crop2(img, target_face.kps, self.input_size[0])
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, self.input_size,
                                     (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        latent = self._normalize_latent(source_face.normed_embedding)
        feeds = {
            self.latent_name: latent,
            self.blob_name: blob
        }
        return feeds, aimg, M

    def get(self, img, target_face, source_face, paste_back=True):
        feeds, aimg, M = self._prepare_inputs(img, target_face, source_face)
        preds = self.session.run(self.output_names, feeds)
        outputs = dict(zip(self.output_names, preds))
        pred = outputs.get('output', preds[0])
        mask = outputs.get('mask', preds[1] if len(preds) > 1 else None)

        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]

        if not paste_back:
            return bgr_fake, M

        target_img = img
        IM = cv2.invertAffineTransform(M)
        bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)

        if mask is not None:
            mask_img = mask[0, 0]
            if mask_img.max() > 1.0:
                mask_img = mask_img / 255.0
            mask_img = np.clip(mask_img, 0.0, 1.0)
            mask_img = cv2.warpAffine(mask_img, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
            mask_img = cv2.GaussianBlur(mask_img, (11, 11), 0)
            mask_img = np.expand_dims(mask_img, axis=2)
        else:
            mask_img = np.full((target_img.shape[0], target_img.shape[1], 1), 1.0, dtype=np.float32)

        fake_merged = mask_img * bgr_fake.astype(np.float32) + (1 - mask_img) * target_img.astype(np.float32)
        return fake_merged.astype(np.uint8)


def patched_get_default_providers():
    return ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']


def patch_insightface(get_default_providers, get_model, faceanalysis_init, faceanalysis_prepare, inswapper_init):
    insightface.model_zoo.model_zoo.get_default_providers = get_default_providers
    insightface.model_zoo.model_zoo.ModelRouter.get_model = get_model
    insightface.app.FaceAnalysis.__init__ = faceanalysis_init
    insightface.app.FaceAnalysis.prepare = faceanalysis_prepare
    insightface.model_zoo.inswapper.INSwapper.__init__ = inswapper_init


original_functions = [patched_get_default_providers, ModelRouter.get_model, FaceAnalysis.__init__, FaceAnalysis.prepare, INSwapper.__init__]
patched_functions = [patched_get_default_providers, patched_get_model, patched_faceanalysis_init, patched_faceanalysis_prepare, patched_inswapper_init]


def apply_logging_patch(console_logging_level):
    if console_logging_level == 0:
        patch_insightface(*patched_functions)
        logger.setLevel(logging.WARNING)
    elif console_logging_level == 1:
        patch_insightface(*patched_functions)
        logger.setLevel(logging.STATUS)
    elif console_logging_level == 2:
        patch_insightface(*original_functions)
        logger.setLevel(logging.INFO)
