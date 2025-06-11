from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=60)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_img = preprocess(frame)
    np.copyto(h_input, input_img.ravel())
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    detections = postprocess(h_output)

    # detections = [((x, y, w, h), conf), ...]
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        x, y, w, h = map(int, track.to_tlwh())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) == 27:
        break
