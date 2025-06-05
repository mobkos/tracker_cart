import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import torch
import numpy as np
import time

HORIZONTAL_FOV = 60.0
HALF_FOV = HORIZONTAL_FOV / 2.0

def compute_angle(cx, frame_width):
    ratio = (cx - frame_width / 2.0) / (frame_width / 2.0)
    return ratio * HALF_FOV

class SimpleKalmanTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                  [0, 1, 0, 1],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.initialized = False

    def update(self, cx, cy):
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        if not self.initialized:
            self.kalman.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
            self.kalman.statePost = self.kalman.statePre
            self.initialized = True
        self.kalman.correct(measurement)
        pred = self.kalman.predict()
        return pred[0], pred[1]  # cx, cy

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True).to(device)
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라 열기 실패")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {W}×{H}, Horizontal FOV: {HORIZONTAL_FOV}")

    tracker = SimpleKalmanTracker()
    last_angle_print_time = 0
    print_interval = 0.5  # 초 단위 출력 주기

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        dets = results.xyxy[0].cpu().numpy()

        person_dets = []
        for *box, conf, cls in dets:
            if int(cls) == 0 and conf > 0.3:  # 사람만
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                person_dets.append((cx, cy))

        if person_dets:
            # 화면 중앙과 가장 가까운 사람 선택
            target = min(person_dets, key=lambda c: abs(c[0] - W / 2.0))
            pred_cx, _ = tracker.update(*target)

            now = time.time()
            if now - last_angle_print_time >= print_interval:
                angle = float(compute_angle(pred_cx, W))
                print(angle)  # 💡 여기서 float 값 그대로 출력
                last_angle_print_time = now

    cap.release()

if __name__ == '__main__':
    main()
