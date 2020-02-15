import cv2
import numpy as np
from models.tem import tem

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes, COLORS):
    label = str(classes[class_id])

    color = COLORS[class_id]
    cv2.rectangle(img, (int(x), int(y)), (int(x_plus_w), int(y_plus_h)), color, 2)

    cv2.putText(img, label, (int(x - 10), int(y - 10)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 1)
    print('Drawing yolo boundbox')

def checkNegative(x, y, w,h):
    if(x < 0):
        x = 0
    if(y < 0):
        y = 0
    if(w < 0):
        w = 0
    if(w < 0):
        w = 0

    return x, y, w, h
def yolo(frame, config, weights, trained, conf):
    image = frame

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(trained, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weights, config)

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    sBoxes = []
    sClass_ids = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)

                x, y, w, h = checkNegative(x, y, w,h)

                confidences.append(float(confidence))
                boxes.append([int(x), int(y), int(w), int(h)])

    print('CONFIDENCE: ', confidence)
    print('Yolo bboxes: ', boxes)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    print("INDICES: ", indices)
    print("Count Objects: ", len(indices))
    boundBox = tem()
    if len(indices) > 0:
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = round(box[0])
            y = round(box[1])
            w = round(box[2])
            h = round(box[3])
            sBoxes.append(boxes[i])
            print(box)
            print(classes[class_ids[i]])
            print(confidences[i])
    else:
        print("Error at index")
        return None

    boundBox.memorize(image, class_ids, confidences, int(x), int(y),
                                        int(x + w), int(y + h), classes, COLORS, indices, sBoxes)

    return boundBox
