import cv2
import numpy as np
import time
# from rknn.api import RKNN
from rknnlite.api import RKNNLite as RKNN

CLASSES = {0: "box", 1: "box_open"}

nmsThresh = 0.45
objectThresh = 0.5

model_h = 640
model_w = 640

color_palette = [
    (0, 255, 0),  # Green for class "box"
    (255, 0, 0)   # Red for class "box_open"
]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class YOLOv8_RKNN:
    def __init__(self, path, conf_thres=nmsThresh, iou_thres=objectThresh):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        self.rknn = RKNN(verbose=False)
        self.rknn.load_rknn(path)
        self.rknn.init_runtime()
    
    def __call__(self, image):
        return self.detect(image)

    def detect(self, image):
        # resized_img, ratio, (dw, dh) = self.letterbox(image)  # padding resize
        resized_img = cv2.resize(image, (model_w, model_h))
        # resized_img = cv2.resize(img, (model_w, model_h), interpolation=cv2.INTER_LINEAR) # direct resize
        input = np.expand_dims(resized_img, axis=0)
        start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[input], data_format="nhwc")

        boxes, scores, class_ids = self.postprocess(image, outputs)
        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")

        return boxes, scores, class_ids

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        return im, ratio, (dw, dh)

    def draw_detections(self, image):
        boxes = self.boxes
        scores = self.scores
        class_ids = self.class_ids
         # Iterate over the selected indices after non-maximum suppression
        for box, score, class_id in zip(boxes, scores, class_ids):
            # Pass ratio and padding to draw_detections
            self.draw_detection(image, box, score, class_id)
        return image

    def draw_detection(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{CLASSES[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img,
            (label_x, label_y - label_height),
            (label_x + label_width, label_y + label_height),
            color,
            cv2.FILLED,
        )

        # Draw the label text on the image
        cv2.putText(
            img,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    def postprocess(self, input_image, outputs):
        img_h, img_w = input_image.shape[:2]
        boxes0 = np.transpose(np.squeeze(outputs[0]))
        scores0 = np.transpose(np.squeeze(outputs[1]))

        if len(scores0.shape) == 1:
            scores0 = np.expand_dims(scores0, axis=1)
        scores = sigmoid(scores0)
        max_scores = np.max(scores, axis=1)  # 多个类别时，最大的分数。
        max_indices = np.argmax(scores, axis=1)

        t = np.where(max_scores >= objectThresh)[0]  # 元组

        boxes = boxes0[t]
        scores = max_scores[t]
        class_ids = max_indices[t]

        # 根据分数从高到低排序
        sorted_indices = np.argsort(scores)[::-1]
        boxes = boxes[sorted_indices]
        scores = scores[sorted_indices]
        class_ids = class_ids[sorted_indices]

        print(boxes)
        print(scores)
        print(class_ids)

        # Get the number of rows in the outputs array
        rows = boxes.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes_ = []
        scores_ = []
        class_ids_ = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = img_w / model_w
        y_factor = img_h / model_h

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = scores[i]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= objectThresh:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = boxes[i]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids_.append(class_id)
                scores_.append(max_score)
                boxes_.append([left, top, width, height])

        print(boxes_)
        print(scores_)
        print(class_ids_)

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(
            boxes_, scores_, score_threshold=objectThresh, nms_threshold=nmsThresh
        )

        boxes = []
        scores = []
        class_ids = []
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            boxes.append(boxes_[i])
            scores.append(scores_[i])
            class_ids.append(class_ids_[i])

        self.boxes = boxes
        self.scores = scores
        self.class_ids = class_ids
        return boxes, scores, class_ids

    def release(self):
        self.rknn.release()


if __name__ == "__main__":
    rknn = YOLOv8_RKNN("../models/best.rknn")

    image = cv2.imread("../models/test_image_3.jpg")

    boxes, scores, class_ids = rknn(image)

    rknn.draw_detections(image)
   
    # 保存结果
    cv2.imwrite("../models/test_image_3_res.jpg", image)

    # 释放
    rknn.release()
