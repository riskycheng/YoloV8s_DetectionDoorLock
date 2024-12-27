import cv2
from BoxDoorUtils import *
import os
from yolov8 import YOLOv8
from yolov8.YOLOv8_RKNN import YOLOv8_RKNN
import sys, getopt
import queue
import threading
from datetime import datetime
from MQTTUtils import MQTTClient
from ConfigManager import ConfigManager
from datetime import datetime, timedelta, timezone
import time

VERSION_CODE = '2024.12.27.v2.0.1'
# default values >>>>>>>>>>>
min_compute_queue_length = 20
min_scale_factor = 0.6
# clear the global queue when continuous frames are empty
clear_global_queue_reaching_empty_det_length = 15
global_current_continuous_empty_count = 0
showUI = True
# expect to run N frame / sec
frames_execute_per_second = 1

# how to handle the treat_half_close_as_close
treat_half_close_as_close = False


# cache door status. true for door-open, false for door-close
global_queue_door_open = []

exit_flag = False

# MQTT related
MQTT_IP_ADDRESS = '192.168.31.170'
MQTT_PORT = 1883
MQTT_TOPIC = 'itvtech/box_open_detection'

ENABLE_MQTT = True
ENABLE_SAVE_OUT_LOGS = True
CLIENT_ID = 'BOX_DET_ALGO'
global_mqtt_client = None
# default values >>>>>>>>>>>


# special design for dual-cameras sync
queueA = queue.Queue()
queueB = queue.Queue()
# Event to signal when frames are ready in both queues
frames_ready_event = threading.Event()

# basic function used for caching frame into queue
def capture_frames(cap, frame_queue, videoAddress, skip_freq, camID = 0, logPath = None):
    
    frameIndex = 0    
    while True:
        if not cap.isOpened():
            print('camera is not opened, retry after 2 seconds...')

            # write it out
            fusedResultInfo = FusedResultInfo()
            fusedResultInfo.camera_idx = camID
            fusedResultInfo.url = videoAddress
            fusedResultInfo.timeStamp = 'offline'
            
            json_str_a = write_file_json(logPath, fusedResultInfo, True, writeOut=ENABLE_SAVE_OUT_LOGS)
            sendMQTTMessage(global_mqtt_client, MQTT_TOPIC, json_str_a, sendOut = ENABLE_MQTT)

            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(videoAddress)
            continue
        try:
            ret, frame = cap.read()
            if not ret or exit_flag:
                print('frame not available, waiting for camera...')
                cap.release()
                continue
            frameIndex += 1
            # print('frameIndex:%d / skip_freq:%d'  %(frameIndex, skip_freq))
            # skip if it is not hitting the tag   
            if frameIndex % skip_freq != 0:
                continue

            frame_queue.put(frame)
            frames_ready_event.set()  # Signal that a frame is ready
        except Exception as e:
            print(e)


def renderCounter(frame):
    frame_height, frame_width = frame.shape[:2]
    textAnchor = (int(frame_width / 2) - 120, int(frame_height - 30))
    # check the queue
    if len(global_queue_door_open) >= min_compute_queue_length:
        class_close_cnt = 0
        class_open_cnt = 0
        for item in global_queue_door_open:
            if item:
                # if any open
                class_open_cnt += 1
        if class_open_cnt >= min_compute_queue_length * min_scale_factor:
            cv2.rectangle(frame, (0, frame_height - 80), (frame_width, frame_height), (0, 153, 255), -1)
            cv2.rectangle(frame, (0, frame_height - 80), (frame_width, frame_height), (0, 120, 255), 2)
            cv2.putText(frame, 'Door Open!', textAnchor, cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)


def renderDualCounter(frame_a, frame_b, timePast):
    
    # Horizontal concatenation
    result_horizontal = cv2.hconcat([frame_a, frame_b])
    if result_horizontal is None:
        print('error: result_horizontal is none')
        return
    res_h, res_w = result_horizontal.shape[:2]
    res_h = int(res_h / 2)
    res_w = int(res_w / 2)

    result_horizontal = cv2.resize(result_horizontal, (res_w, res_h))

    # check the queue
    # CLASSES = {0: "door_half_close", 1: "door_close", 2: "door_open"}
    if len(global_queue_door_open) >= min_compute_queue_length:
        class_open_cnt = 0
        for item in global_queue_door_open:
            if item:
                # if any open
                class_open_cnt += 1

        if class_open_cnt >= min_compute_queue_length * min_scale_factor:
           # render the rectangle
           cv2.rectangle(result_horizontal, (0, res_h - 40), (res_w, res_h), (0, 153, 255), -1)
           cv2.rectangle(result_horizontal, (0, res_h - 40), (res_w, res_h), (0, 120, 255), 2)
           textAnchor = (int(res_w / 2) - 60, int(res_h - 10))
           cv2.putText(result_horizontal, 'Door Open!', textAnchor, cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

    timePastTxt = 'Time-past: ' + (timePast)
    (label_width, label_height), _ = cv2.getTextSize(
                timePastTxt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    cv2.rectangle(result_horizontal, (0, 0), (6 + label_width, 6 + label_height), (0, 153, 255), -1)
    
    textAnchor = (0, 0 + label_height)

    cv2.putText(result_horizontal, timePastTxt, textAnchor, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            
    return result_horizontal


def startDualExe(videoAddressA, videoAddressB, runNPU=True):
    global global_current_continuous_empty_count, exit_flag, showUI
    
    # dual cameras
    capA = cv2.VideoCapture(videoAddressA)
    capB = cv2.VideoCapture(videoAddressB)

    # expect both cameras have the same FPS
    video_fps = capA.get(cv2.CAP_PROP_FPS)

    if video_fps > 30:
        video_fps = 25

    # private skipping frequency
    skip_freq = int(video_fps / frames_execute_per_second)
    print('video fps :' + str(video_fps) + ' skip_freq : ' + str(skip_freq))


    rknn_model_path = "./models/best.rknn"
    onnx_model_path = './models/best.onnx'
    if runNPU:
        yolov8_detector = YOLOv8_RKNN(rknn_model_path)
    else:
        yolov8_detector = YOLOv8(onnx_model_path)

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
    threadA = threading.Thread(target=capture_frames, args=(capA, queueA, videoAddressA, video_fps, 0, logPath))
    threadB = threading.Thread(target=capture_frames, args=(capB, queueB, videoAddressB, video_fps, 1, logPath))
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

        # measure queues
        queueA_len = queueA.qsize()
        queueB_len = queueB.qsize()

        if queueA_len > 5 or queueB_len > 5:
            queueA.queue.clear()
            queueB.queue.clear()

        # Get the current system time
        current_time = datetime.now(timezone(timedelta(hours=8)))  # Assuming UTC+8 timezone
        formatted_time_with_ms = current_time.isoformat(timespec='milliseconds')

        fusedResultInfo_A = FusedResultInfo()
        fusedResultInfo_A.camera_idx = 0
        fusedResultInfo_A.url = videoAddressA
        fusedResultInfo_A.timeStamp = formatted_time_with_ms
        
        fusedResultInfo_B = FusedResultInfo()
        fusedResultInfo_B.camera_idx = 1
        fusedResultInfo_B.url = videoAddressB
        fusedResultInfo_B.timeStamp = formatted_time_with_ms


        # start processing
        boxes_a, scores_a, class_ids_a = yolov8_detector(frame_a)
        # show Box with TimeStamp 
        # combined_img_a = yolov8_detector.draw_detections(frame_a, formatted_time_with_ms)
        # show Box without TimeStamp 
        combined_img_a = yolov8_detector.draw_detections(frame_a)
        boxes_b, scores_b, class_ids_b = yolov8_detector(frame_b)
        # show Box with TimeStamp 
        # combined_img_b = yolov8_detector.draw_detections(frame_b, formatted_time_with_ms)
        # show Box without TimeStamp 
        combined_img_b = yolov8_detector.draw_detections(frame_b)

        executedFrameCount += 1
        
        if len(boxes_a) == 0 and len(boxes_b) == 0:
            global_current_continuous_empty_count += 1
            if global_current_continuous_empty_count >= clear_global_queue_reaching_empty_det_length:
                global_queue_door_open.clear()
        else:
            global_current_continuous_empty_count = 0
        
        # compute the counter - A
        # CLASSES = {0: "door_half_close", 1: "door_close", 2: "door_open"}
        anyDoorOpen = False
        for box, score, class_id in zip(boxes_a, scores_a, class_ids_a):
            if class_id == 2: # indicates door_open
                anyDoorOpen = True
            elif class_id == 0 and not treat_half_close_as_close : # indicates door-half-close
                anyDoorOpen = True
            else:
                anyDoorOpen = False
            
            # assign to the structured data info
            doorInfo = DoorDetResultInfo()
            doorInfo.label = int(class_id)
            doorInfo.conf = float("{:.2f}".format(score))
            # box in xywh format
            boxInfo = object_rect(x=int(box[0]), y=int(box[1]), width=int(box[2]), height=int(box[3]))
            doorInfo.boundingBox = boxInfo
            fusedResultInfo_A.doorInfoArray.append(doorInfo)

        # compute the counter - B
        # CLASSES = {0: "door_half_close", 1: "door_close", 2: "door_open"}
        for box, score, class_id in zip(boxes_b, scores_b, class_ids_b):
            if class_id == 2: # indicates door-open
                anyDoorOpen = True
            elif class_id == 0 and not treat_half_close_as_close:
                anyDoorOpen = True
            else:
                anyDoorOpen = False
            
            # assign to the structured data info
            doorInfo = DoorDetResultInfo()
            doorInfo.label = int(class_id)
            doorInfo.conf = float("{:.2f}".format(score))
            # box in xyxy format
            boxInfo = object_rect(x=int(box[0]), y=int(box[1]), width=int(box[2]), height=int(box[3]))
            doorInfo.boundingBox = boxInfo
            fusedResultInfo_B.doorInfoArray.append(doorInfo)
        
        if anyDoorOpen:
            global_queue_door_open.append(True) # any door open
        else:
            global_queue_door_open.append(False) # door all close

        # calculate the actual pastime in seconds
        pastTime_second = executedFrameCount / frames_execute_per_second
        final_concat_img = renderDualCounter(combined_img_a, combined_img_b, convert_seconds_to_ddhhmmss(pastTime_second))

        # add timeStamp in the left bottom region
        timeStampTxt = formatted_time_with_ms
        # Calculate the dimensions of the label text
        (timeStampTxt_width, timeStampTxt_height), _ = cv2.getTextSize(timeStampTxt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        final_concat_img_height, final_concat_img_width = final_concat_img.shape[:2]
        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            final_concat_img,
            (0, final_concat_img_height - 50 - timeStampTxt_height),
            (timeStampTxt_width, final_concat_img_height - 50),
            (0, 0, 255),
            cv2.FILLED,
        )
        # draw the label text on the image standing for the timeStamp
        cv2.putText(
            final_concat_img,
            timeStampTxt,
            (0, final_concat_img_height - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
            )
        
        if final_concat_img is None:
            print('error: final_concat_img is none')
            continue
        if showUI:
            cv2.imshow('Container Door Detection System @itvtech' + VERSION_CODE, final_concat_img)
            cv2.waitKey(1)

        json_str_a = write_file_json(logPath, fusedResultInfo_A, True, writeOut=ENABLE_SAVE_OUT_LOGS)
        sendMQTTMessage(global_mqtt_client, MQTT_TOPIC, json_str_a, sendOut = ENABLE_MQTT)
        json_str_b = write_file_json(logPath, fusedResultInfo_B, True, writeOut=ENABLE_SAVE_OUT_LOGS)
        sendMQTTMessage(global_mqtt_client, MQTT_TOPIC, json_str_b, sendOut = ENABLE_MQTT)

    threadA.join()
    threadB.join()

    capA.release()
    capB.release()
  

if '__main__' == __name__:

    if not os.path.exists('config.json'):
        print('Error: Please Ensure config.json exist in the same location!')
        sys.exit(1)
    
    config_manager = ConfigManager('config.json')
    
    # get the configs
    RUN_ON_NPU = config_manager.get('run_on_npu')
    MQTT_IP_ADDRESS = config_manager.get('host_address')
    MQTT_PORT = config_manager.get('host_port')
    MQTT_TOPIC = config_manager.get('topic')
    ENABLE_MQTT = config_manager.get('enable_MQTT')
    ENABLE_SAVE_OUT_LOGS = config_manager.get('save_out_logs')
    RTSP_A = config_manager.get('rtsp_address_a')
    RTSP_B = config_manager.get('rtsp_address_b')
    frames_execute_per_second = config_manager.get('frames_execute_per_second')
    min_compute_queue_length = config_manager.get('min_compute_queue_length')
    min_scale_factor = config_manager.get('min_scale_factor')
    clear_global_queue_reaching_empty_det_length = config_manager.get('clear_global_queue_reaching_empty_det_length')
    treat_half_close_as_close = config_manager.get('treat_half_close_as_close')
    showUI = config_manager.get('show_UI')

    print('Configurated: \n',
          '\tRUN_ON_NPU:%s \n' %('NPU' if RUN_ON_NPU else 'CPU'),
          '\tMQTT-ADDR:%s:%d @topic:%s\n' %(MQTT_IP_ADDRESS, MQTT_PORT, MQTT_TOPIC),
          '\tRTSP-A:%s \n' %RTSP_A,
          '\tRTSP-B:%s \n' %RTSP_B,
          '\tframes_execute_per_second:%d \n' %frames_execute_per_second,
          '\tENABLE_MQTT:%s \n' %('True' if ENABLE_MQTT else 'False'),
          '\tENABLE_SAVE_OUT_LOGS:%s \n' %('True' if ENABLE_SAVE_OUT_LOGS else 'False'),
          '\tmin_compute_queue_length:%d \n' %min_compute_queue_length,
          '\tmin_scale_factor:%.2f \n' %min_scale_factor,
          '\tclear_global_queue_reaching_empty_det_length:%.2f \n' %clear_global_queue_reaching_empty_det_length,
          '\tshowUI:%s \n' %('True' if showUI else 'False'),
          '\ttreat_half_close_as_close:%s \n' % ('True' if treat_half_close_as_close else 'False'),
          )

    # start the MQTT connection
    if ENABLE_MQTT:
        global_mqtt_client = MQTTClient(MQTT_IP_ADDRESS, MQTT_PORT, CLIENT_ID)
        global_mqtt_client.connect()
    
    # start the processing engine
    print('Start Processing, press \'q\' to exit \n')
    startDualExe(RTSP_A, RTSP_B, RUN_ON_NPU)