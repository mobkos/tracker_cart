import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def preprocess(frame, shape=(640, 640)):
    image = cv2.resize(frame, shape)
    image = image[:, :, ::-1].transpose(2, 0, 1) / 255.0
    image = np.ascontiguousarray(image, dtype=np.float32)
    return image

def postprocess(outputs, conf_thresh=0.3):
    # YOLOv5 postprocessing 생략 — ONNX 추론 결과에 따라 파싱
    return []

engine = load_engine("yolov5n.trt")
context = engine.create_execution_context()
h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
