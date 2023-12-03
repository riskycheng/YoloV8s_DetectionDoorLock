import cv2
from BoxDoorUntils import *

from yolov8 import YOLOv8
import sys, getopt
import queue
import threading
from datetime import datetime

min_compute_queue_length = 20
min_scale_factor = 0.6
# clear the global queue when continuous frames are empty
clear_global_queue_reaching_empty_det_length = 15
global_current_continuous_empty_count = 0
global_queue = []

exit_flag = False

# special design for dual-cameras sync
queueA = queue.Queue()
queueB = queue.Queue()
# Event to signal when frames are ready in both queues
frames_ready_event = threading.Event()

# basic function used for caching frame into queue
def capture_frames(cap, frame_queue):
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret or exit_flag:
                break
            frame_queue.put(frame)
            frames_ready_event.set()  # Signal that a frame is ready
        except Exception as e:
            print(e)


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


def renderDualCounter(frame_a, frame_b):
    
    # Horizontal concatenation
    result_horizontal = cv2.hconcat([frame_a, frame_b])
    res_h, res_w = result_horizontal.shape[:2]
    res_h = int(res_h / 4)
    res_w = int(res_w / 4)
    print(res_h, res_w)
    result_horizontal = cv2.resize(result_horizontal, (res_w, res_h))

    # check the queue
    if len(global_queue) >= min_compute_queue_length:
        class_open_cnt = 0
        for item in global_queue:
            if item == 1:
                # if any open
                class_open_cnt += 1
        
        if class_open_cnt >= min_compute_queue_length * min_scale_factor:
           # render the rectangle
           cv2.rectangle(result_horizontal, (0, res_h - 40), (res_w, res_h), (0, 153, 255), -1)
           cv2.rectangle(result_horizontal, (0, res_h - 40), (res_w, res_h), (0, 120, 255), 2)
           textAnchor = (int(res_w / 2) - 60, int(res_h - 10))
           cv2.putText(result_horizontal, 'Door Open!', textAnchor, cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
        
    return result_horizontal


def startSingleExe(videoAddress):
    global global_current_continuous_empty_count
    cap = cv2.VideoCapture(videoAddress)
    model_path = "./models/best.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

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

        cv2.imshow("single-RTSP", combined_img)




def startDualExe(videoAddressA, videoAddressB):
    global global_current_continuous_empty_count, exit_flag
    
    # dual cameras
    capA = cv2.VideoCapture(videoAddressA)
    capB = cv2.VideoCapture(videoAddressB)

    model_path = "./models/best.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

    # expect both cameras have the same FPS
    video_fps = capA.get(cv2.CAP_PROP_FPS)
    
    # tune this value below
    frames_execute_per_second = 1

    # private skipping frequency
    skip_freq = int(video_fps / frames_execute_per_second)

    print('video fps :' + str(video_fps) + ' skip_freq : ' + str(skip_freq))

    frameIndex = 0
    executedFrameCount = 0

    if not capA.isOpened() or not capB.isOpened():
        print('either RTSP-A or RTSP-B is not available!')
        return
    
    # save out to log
    # Get the current system time
    current_time = datetime.now()
    time_str = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    logPath = './log_' + time_str + '.txt'

    # Start separate threads to capture frames for each RTSP stream
    threadA = threading.Thread(target=capture_frames, args=(capA, queueA))
    threadB = threading.Thread(target=capture_frames, args=(capB, queueB))
    threadA.start()
    threadB.start()


    while not exit_flag:
        # Press key q to stop
        if cv2.waitKey(1) == ord('q'):
            exit_flag = True
            break
        
        frames_ready_event.wait()  # Wait until frames are ready
        frame_a = queueA.get()
        frame_b = queueB.get()
        frames_ready_event.clear()  # Reset the event
        frameIndex += 1
        if frameIndex % skip_freq != 0:
            continue

        # measure queues
        queueA_len = queueA.qsize()
        queueB_len = queueB.qsize()
        print('queueA_len:%d, queueB_len:%d' %(queueA_len, queueB_len))

        if queueA_len > 5 or queueB_len > 5:
            queueA.queue.clear()
            queueB.queue.clear()

        # Get the current system time
        current_time = datetime.now()
        time_str = current_time.strftime("%Y_%m_%d_%H_%M_%S")

        
        fusedResultInfo_A = FusedResultInfo()
        fusedResultInfo_A.camera_idx = addr1
        fusedResultInfo_A.timeStamp = time_str
        
        fusedResultInfo_B = FusedResultInfo()
        fusedResultInfo_B.camera_idx = addr2
        fusedResultInfo_B.timeStamp = time_str


        # start processing
        boxes_a, scores_a, class_ids_a = yolov8_detector(frame_a)
        combined_img_a = yolov8_detector.draw_detections(frame_a)
        boxes_b, scores_b, class_ids_b = yolov8_detector(frame_b)
        combined_img_b = yolov8_detector.draw_detections(frame_b)

        executedFrameCount += 1
        print('executed frames:' + str(executedFrameCount))
        
        if len(boxes_a) == 0 and len(boxes_b) == 0:
            global_current_continuous_empty_count += 1
            if global_current_continuous_empty_count >= clear_global_queue_reaching_empty_det_length:
                global_queue.clear()
        else:
            global_current_continuous_empty_count = 0
        
        # compute the counter - A
        anyDoorOpen = False
        for box, score, class_id in zip(boxes_a, scores_a, class_ids_a):
            if class_id == 1: # indicates door-open
                anyDoorOpen = True
            
            # assign to the structured data info
            doorInfo = DoorDetResultInfo()
            doorInfo.label = class_id
            doorInfo.conf = score
            # box in xyxy format
            boxInfo = object_rect(x=box[0], y=box[1], width=box[2] - box[0], height=box[3] - box[1])
            doorInfo.boundingBox = boxInfo
            fusedResultInfo_A.doorInfoArray.append(doorInfo)
        
        for box, score, class_id in zip(boxes_b, scores_b, class_ids_b):
            if class_id == 1: # indicates door-open
                anyDoorOpen = True
            
            # assign to the structured data info
            doorInfo = DoorDetResultInfo()
            doorInfo.label = class_id
            doorInfo.conf = score
            # box in xyxy format
            boxInfo = object_rect(x=box[0], y=box[1], width=box[2] - box[0], height=box[3] - box[1])
            doorInfo.boundingBox = boxInfo
            fusedResultInfo_B.doorInfoArray.append(doorInfo)
        
        if anyDoorOpen:
            global_queue.append(1) # any door open
        else:
            global_queue.append(0) # door all close

        final_concat_img = renderDualCounter(combined_img_a, combined_img_b)

        cv2.imshow("Dual-RTSP", final_concat_img)
        cv2.waitKey(1)

        write_file_json(logPath, fusedResultInfo_A, True)
        write_file_json(logPath, fusedResultInfo_B, True)

    threadA.join()
    threadB.join()

    capA.release()
    capB.release()
  

# rtsp://admin:itv12345@192.168.1.187:554/Streaming/channels/101
# rtsp://admin:itv12345@192.168.1.187:554/Streaming/channels/201
if '__main__' == __name__:
    argv = sys.argv
    argc = len(argv)
        
    if argc == 2:
        addr = argv[1]
        startSingleExe(addr)
    elif argc == 3:
        addr1 = argv[1]
        addr2 = argv[2]
        startDualExe(addr1, addr2)
    else:
        print('python xx.py rtsp-addr or python xx.py rtsp-addr-a rtsp-addr-b')
