import json
import numpy as np

class object_rect:
    def __init__(self, x=None, y=None, width=None, height=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

class DoorDet_config:
    def __init__(self, det_threshold=None, compute_every_frames=None, sync_results_frame=None, shared_mem_id=None, shared_sem_id=None, sync_waiting_shared_memory_consumed=None):
        self.det_threshold = det_threshold
        self.compute_every_frames = compute_every_frames
        self.sync_results_frame = sync_results_frame
        self.shared_mem_id = shared_mem_id
        self.shared_sem_id = shared_sem_id
        self.sync_waiting_shared_memory_consumed = sync_waiting_shared_memory_consumed


class DoorDetResultInfo:
    def __init__(self, bounding_box=None, label=None, conf=None):
        self.boundingBox = bounding_box
        self.label = label
        self.conf = conf

class FusedResultInfo:
    def __init__(self, door_info_array=None, camera_idx=None, time_stamp=None):
        self.doorInfoArray = door_info_array or [] # array of DoorDetResultInfo
        self.camera_idx = camera_idx
        self.timeStamp = time_stamp



def convert_to_python_type(obj):
    if isinstance(obj, (np.floating, np.int64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def write_file_json(file_path, info, append, writeOut = True):
    root = {
        "camera_idx": info.camera_idx,
        "timeStamp": info.timeStamp,
        "doors": [],
        "anyDoorOpen": any(item.label > 0 for item in info.doorInfoArray)
    }

    for item in info.doorInfoArray:
        box = {
            "x": item.boundingBox.x,
            "y": item.boundingBox.y,
            "width": item.boundingBox.width,
            "height": item.boundingBox.height,
            "status": item.label,
            "confidence": item.conf
        }
        root["doors"].append(box)

    # Convert the Python data structure to JSON string with newline
    json_data = json.dumps(root, indent=4, default=convert_to_python_type) + "\n"

    # Write to file
    if write_file_json:
        mode = "a" if append else "w"
        with open(file_path, mode) as file:
            # Ensure that the new JSON data starts on a new line when appending
            if append:
                file.write("\n")
            file.write(json_data)

    return json_data



def sendMQTTMessage(mqtt_client, topic, json_message, sendOut = True):
    if not sendMQTTMessage:
        return
    if mqtt_client.connected:
        mqtt_client.publish(topic, json_message)
    else:
        print('error: mqtt service is not connected!')