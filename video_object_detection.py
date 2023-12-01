import cv2

from yolov8 import YOLOv8
import sys, getopt

min_compute_queue_length = 20
min_scale_factor = 0.6
# clear the global queue when continuous frames are empty
clear_global_queue_reaching_empty_det_length = 15
global_current_continuous_empty_count = 0
global_queue = []


def renderCounter(frame):
    frame_height, frame_width = frame.shape[:2]
    textAnchor = (int(frame_width / 2) - 120, int(frame_height - 30))
    # check the queue
    if len(global_queue) >= min_compute_queue_length:
        class_close_cnt = 0
        class_open_cnt = 0
        for item in global_queue:
            if item == 1:
                # if any open
                class_open_cnt += 1
        if class_open_cnt >= min_compute_queue_length * min_scale_factor:
            cv2.rectangle(frame, (0, frame_height - 80), (frame_width, frame_height), (0, 153, 255), -1)
            cv2.rectangle(frame, (0, frame_height - 80), (frame_width, frame_height), (0, 120, 255), 2)
            cv2.putText(frame, 'Door Open!', textAnchor, cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)


def startExe(videoAddress):
    global global_current_continuous_empty_count
    cap = cv2.VideoCapture(videoAddress)
    model_path = "./models/best.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # tune this value below
    frames_execute_per_second = 1

    # private skipping frequency
    skip_freq = int(video_fps / frames_execute_per_second)

    print('video fps :' + str(video_fps) + ' skip_freq : ' + str(skip_freq))

    frameIndex = 0
    executedFrameCount = 0
    while cap.isOpened():

        # Press key q to stop
        if cv2.waitKey(1) == ord('q'):
            break

        try:
            # Read frame from the video
            ret, frame = cap.read()
            frameIndex += 1
            if frameIndex % skip_freq != 0:
                continue
            if not ret:
                break
        except Exception as e:
            print(e)
            continue

        # Update object localizer
        boxes, scores, class_ids = yolov8_detector(frame)
        executedFrameCount += 1
        print('executed frames:' + str(executedFrameCount))

        if len(boxes) == 0:
            global_current_continuous_empty_count += 1
            if global_current_continuous_empty_count >= clear_global_queue_reaching_empty_det_length:
                global_queue.clear()
        else:
            global_current_continuous_empty_count = 0

        # compute the counter
        for box, score, class_id in zip(boxes, scores, class_ids):
            global_queue.append(class_id)

        combined_img = yolov8_detector.draw_detections(frame)

        # render the bottom rectangle
        renderCounter(combined_img)

        cv2.imshow("Detected Objects", combined_img)

# rtsp://admin:itv12345@192.168.1.187:554/Streaming/channels/101
# rtsp://admin:itv12345@192.168.1.187:554/Streaming/channels/201
if '__main__' == __name__:
    argv = sys.argv
    argc = len(argv)
    if argc != 2:
        print('python xx.py rtsp-addr')
    else:
        addr = argv[1]
        startExe(addr)
