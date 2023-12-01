import cv2

from yolov8 import YOLOv8

# Initialize yolov8 object detector
model_path = "./models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

# Read image
img = cv2.imread('./models/test_image.jpg')
cv2.imshow("Detected Objects", img)
# Detect Objects
boxes, scores, class_ids = yolov8_detector(img)

# Draw detections
combined_img = yolov8_detector.draw_detections(img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("./models/detected_objects.jpg", combined_img)
cv2.waitKey(0)
