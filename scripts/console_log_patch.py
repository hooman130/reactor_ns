import os.path as osp
import glob
import logging
import numpy as np
import insightface
from insightface.model_zoo.model_zoo import ModelRouter, PickableInferenceSession, get_default_providers
from insightface.model_zoo.retinaface import RetinaFace
from insightface.model_zoo.landmark import Landmark
from insightface.model_zoo.attribute import Attribute
from insightface.model_zoo.inswapper import INSwapper
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.app import FaceAnalysis
from insightface.utils import DEFAULT_MP_NAME, ensure_available
from insightface.model_zoo import model_zoo
import onnxruntime
import onnx
from onnx import numpy_helper
from scripts.reactor_logger import logger


def patched_get_model(self, **kwargs):
    session = PickableInferenceSession(self.onnx_file, **kwargs)
    inputs = session.get_inputs()
    input_cfg = inputs[0]
    input_shape = input_cfg.shape
    input_height = input_shape[2] if len(input_shape) > 2 else None
    input_width = input_shape[3] if len(input_shape) > 3 else None
    outputs = session.get_outputs()

    # The stock router indexes spatial dims directly (input_shape[2]/[3]) which fails
    # for models like hyperswap_256 that publish only batch/channel dims. We guard
    # every access and fall back to INSwapper when the shape is incomplete so the
    # model still loads without editing insightface itself.
    if len(outputs) >= 5:
        return RetinaFace(model_file=self.onnx_file, session=session)
    elif input_height == 192 and input_width == 192:
        return Landmark(model_file=self.onnx_file, session=session)
    elif input_height == 96 and input_width == 96:
        return Attribute(model_file=self.onnx_file, session=session)
    elif len(inputs) == 2:
        # Some swapped-face models (e.g., Hyperswap 256) do not expose spatial dims in
        # the ONNX session input metadata, so gracefully fall back to INSwapper if we
        # cannot infer a more specific type. Default to INSwapper for all two-input
        # models to avoid returning None for unknown variants.
        if input_height is not None and input_height == input_width and input_height in [128, 256]:
            return INSwapper(model_file=self.onnx_file, session=session)
        return INSwapper(model_file=self.onnx_file, session=session)
    elif input_height is not None and input_height == input_width and input_height >= 112 and input_height % 16 == 0:
        return ArcFaceONNX(model_file=self.onnx_file, session=session)
    else:
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
    if latent_dim is None:
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

        if arr.shape[0] == latent_dim:
            best = arr
            break
        if arr.shape[1] == latent_dim:
            best = arr.T
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
    return latent_dim


def patched_inswapper_init(self, model_file=None, session=None):
    self.model_file = model_file
    self.session = session
    model = onnx.load(self.model_file)
    graph = model.graph
    inputs = self.session.get_inputs() if self.session is not None else None
    latent_dim = _infer_latent_dim(inputs, graph) if inputs is not None else None
    self.emap = _pick_emap(graph, latent_dim)
    self.input_mean = 0.0
    self.input_std = 255.0
    if self.session is None:
        self.session = onnxruntime.InferenceSession(self.model_file, None)
        inputs = self.session.get_inputs()
        if latent_dim is None:
            latent_dim = _infer_latent_dim(inputs, graph)
            self.emap = _pick_emap(graph, latent_dim)
    inputs = self.session.get_inputs()
    self.input_names = []
    for inp in inputs:
        self.input_names.append(inp.name)
    outputs = self.session.get_outputs()
    output_names = [out.name for out in outputs]
    if not output_names:
        graph_output_names = [out.name for out in graph.output]
        raise ValueError(
            f"No output nodes found in ONNX model {self.model_file}. Session outputs: {output_names}, "
            f"graph outputs: {graph_output_names}"
        )
    if len(output_names) > 1:
        logger.warning(
            "Expected a single ONNX output for INSwapper, but found %s. Using the first output '%s'.",
            len(output_names),
            output_names[0],
        )
        output_names = [output_names[0]]
    self.output_names = output_names
    input_cfg = inputs[0]
    input_shape = input_cfg.shape
    self.input_shape = input_shape
    if len(input_shape) >= 4 and input_shape[2] is not None and input_shape[3] is not None:
        self.input_size = tuple(input_shape[2:4][::-1])
    else:
        inferred_size = None
        # Attempt to infer static dimensions from the ONNX graph definition first.
        for value_info in graph.input:
            if value_info.name == input_cfg.name and value_info.type.HasField("tensor_type"):
                tensor_shape = value_info.type.tensor_type.shape
                dims = []
                for dim in tensor_shape.dim:
                    if dim.HasField("dim_value"):
                        dims.append(dim.dim_value)
                    elif dim.HasField("dim_param"):
                        try:
                            dims.append(int(dim.dim_param))
                        except (TypeError, ValueError):
                            dims.append(None)
                    else:
                        dims.append(None)

                if len(dims) >= 4 and dims[2] and dims[3]:
                    inferred_size = (dims[3], dims[2])
                    break

        # Fall back to a sensible default if the model omits spatial dims (e.g., Hyperswap 256).
        if inferred_size is None:
            model_name = osp.basename(self.model_file).lower() if self.model_file else ""
            if "256" in model_name:
                inferred_size = (256, 256)
            else:
                inferred_size = (128, 128)
            logger.warning(
                "Model %s does not expose spatial input dims; defaulting INSwapper input size to %sx%s.",
                model_name or "unknown",
                inferred_size[0],
                inferred_size[1],
            )

        self.input_size = inferred_size


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
    # Always apply the patched insightface hooks so model routing and INSwapper
    # initialization remain resilient (required for models like hyperswap_256),
    # while only the logger verbosity changes per level.
    patch_insightface(*patched_functions)

    if console_logging_level == 0:
        logger.setLevel(logging.WARNING)
    elif console_logging_level == 1:
        logger.setLevel(logging.STATUS)
    elif console_logging_level == 2:
        logger.setLevel(logging.INFO)
