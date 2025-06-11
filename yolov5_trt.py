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
    # Resize + BGR->RGB + transpose + normalize + contiguous
    image = cv2.resize(frame, shape)
    image = image[:, :, ::-1].transpose(2, 0, 1) / 255.0
    image = np.ascontiguousarray(image, dtype=np.float32)
    return image

def postprocess(output, conf_thresh=0.3):
    output = output.reshape(-1, 85)  # YOLOv5 output shape
    boxes, scores, class_ids = [], [], []

    for det in output:
        conf = det[4]
        if conf < conf_thresh:
            continue
        class_score = det[5:]
        cls_id = np.argmax(class_score)
        score = class_score[cls_id] * conf
        if score < conf_thresh:
            continue

        cx, cy, w, h = det[0], det[1], det[2], det[3]
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        boxes.append([x1, y1, x2, y2])
        scores.append(float(score))
        class_ids.append(int(cls_id))

    return boxes, scores, class_ids

class YoLov5TRT:
    def __init__(self, engine_path):
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.h_input, self.d_input, self.h_output, self.d_output, self.stream = allocate_buffers(self.engine)

    def infer(self, frame):
        img = preprocess(frame)
        np.copyto(self.h_input, img.ravel())

        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return postprocess(self.h_output)
