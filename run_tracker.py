import cv2
import time
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from yolov5_trt import YoLov5TRT  # 위 파일 import

HORIZONTAL_FOV = 60.0
HALF_FOV = HORIZONTAL_FOV / 2.0

def compute_angle(cx, frame_width):
    ratio = (cx - frame_width / 2.0) / (frame_width / 2.0)
    return ratio * HALF_FOV

def main():
    engine_path = "yolov5n.trt"  # TensorRT 엔진 파일 경로
    detector = YoLov5TRT(engine_path)
    tracker = DeepSort(max_age=60, embedder='mobilenet')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 해상도 640x480
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
    print_interval = 0.3  # 프레임 간격 줄임 (0.3초)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, scores, class_ids = detector.infer(frame)

        dets_for_sort = []
        for box, score, cls in zip(boxes, scores, class_ids):
            if cls == 0 and score > 0.3:  # 사람 클래스만 추적
                x1, y1, x2, y2 = box
                dets_for_sort.append(((x1, y1, x2 - x1, y2 - y1), score))

        tracks = tracker.update_tracks(dets_for_sort, frame=frame)

        now = time.time()

        if user_id is not None and now - last_user_seen > 3:
            print("사용자 분실, 재지정 시도")
            user_id = None
            user_announced = False

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
                user_announced = False

        if user_id is not None and not user_announced:
            print(f"사용자 확인됨: ID {user_id}")
            user_announced = True

        for track in tracks:
            if not track.is_confirmed() or track.track_id != user_id:
                continue

            x, y, w, h = map(int, track.to_tlwh())
            cx = x + w / 2.0
            angle = compute_angle(cx, W)

            last_user_seen = now

            if now - last_angle_print_time >= print_interval:
                print(f"사용자 각도: {angle:+.1f}°")
                last_angle_print_time = now

            break

        # ESC 키 누르면 종료
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
