import cv2 as cv
import numpy as np
import os

yolo_weights = os.path.join(os.getcwd(),"yolo/", "yolov3.weights")
yolo_cfg = os.path.join(os.getcwd(),"yolo/", "yolov3.cfg")
classes_file = os.path.join(os.getcwd(),"yolo/", "coco.names")


net = cv.dnn.readNet(yolo_weights, yolo_cfg)

with open(classes_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]


layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

def detect_objects(image_path, conf_threshold=0.5, nms_threshold=0.4):
    img = cv.imread(image_path)
    height, width, channels = img.shape

    blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    detected_objects = []
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        detected_objects.append({"label": label, "confidence": confidences[i], "box": [x, y, w, h]})
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(img, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    result_path = os.path.join("uploads", "processed_" + os.path.basename(image_path))
    cv.imwrite(result_path, img)

    return result_path, detected_objects
