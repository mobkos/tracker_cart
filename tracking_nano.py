import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import torch
import numpy as np
import time
from deep_sort_realtime.deepsort_tracker import DeepSort

HORIZONTAL_FOV = 60.0
HALF_FOV = HORIZONTAL_FOV / 2.0

def compute_angle(cx, frame_width):
    ratio = (cx - frame_width / 2.0) / (frame_width / 2.0)
    return ratio * HALF_FOV

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True).to(device)
    model.eval()

    tracker = DeepSort(max_age=60, embedder='mobilenet')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라 열기 실패")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {W}×{H}, Horizontal FOV: {HORIZONTAL_FOV}")

    user_id = None
    user_announced = False
    last_user_seen = time.time()
    last_angle_print_time = 0
    print_interval = 0.5  # 출력 간격

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        dets = results.xyxy[0].cpu().numpy()

        dets_for_sort = []
        for *box, conf, cls in dets:
            if int(cls) == 0 and conf > 0.3:  # 사람만
                x1, y1, x2, y2 = map(int, box)
                dets_for_sort.append(((x1, y1, x2 - x1, y2 - y1), float(conf)))

        tracks = tracker.update_tracks(dets_for_sort, frame=frame)

        now = time.time()

        # 사용자 미확인 시간 초과 시 → 재지정
        if user_id is not None and now - last_user_seen > 3:
            print("사용자 분실, 재지정 시도")
            user_id = None
            user_announced = False

        # 사용자 미지정 상태면 화면 중앙에 가까운 사람으로 지정
        if user_id is None:
            min_dist = float('inf')
            best_id = None
            for track in tracks:
                if not track.is_confirmed():
                    continue
                x, y, w, h = map(int, track.to_tlwh())
                cx = x + w / 2.0
                dist = abs(cx - W / 2.0)
                if dist < min_dist:
                    min_dist = dist
                    best_id = track.track_id

            if best_id is not None:
                user_id = best_id
                last_user_seen = now
                user_announced = False  # 다시 print하게 하기

        # 사용자 최초 확인 시 print
        if user_id is not None and not user_announced:
            print(f"사용자 확인됨: ID {user_id}")
            user_announced = True

        # 사용자 추적
        for track in tracks:
            if not track.is_confirmed() or track.track_id != user_id:
                continue

            x, y, w, h = map(int, track.to_tlwh())
            cx = x + w / 2.0
            angle = compute_angle(cx, W)

            last_user_seen = now

            if now - last_angle_print_time >= print_interval:
                print(f"{angle:+.1f}")
                last_angle_print_time = now

            break  # 사용자 1명만 추적

    cap.release()

if __name__ == '__main__':
    main()
