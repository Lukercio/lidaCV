import cv2
import numpy as np
import csv

class tem:
    def __init__(self):
        pass

    def set_image(self, image):
        self.image = image

    def set_class_ids(self, class_ids):
        self.class_ids = class_ids

    def set_confidences(self, confidences):
        self.confidences = confidences

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def set_xw(self, xw):
        self.xw = xw

    def set_yh(self, yh):
        self.yh = yh

    def set_classes(self, classes):
        self.classes = classes

    def set_COLOR(self, COLOR):
        self.COLOR = COLOR

    def set_indices(self, indices):
        self.indices = indices

    def set_boxes(self, boxes):
        self.boxes = boxes

    def set_box(self, i, boxes):
        self.boxes[i] = boxes

    def memorize(self, image, class_ids, confidences, x, y, xw, yh, classes, COLOR, indices, boxes):
        self.set_image(image)
        self.set_class_ids(class_ids)
        self.set_confidences(confidences)
        self.set_x(x)
        self.set_y(y)
        self.set_xw(xw)
        self.set_yh(yh)
        self.set_classes(classes)
        self.set_COLOR(COLOR)
        self.set_indices(indices)
        self.set_boxes(boxes)

    # Save information (m) to a csv file (file) and an output dir
    def persistMemory(info, file, output):
        with open(output + file, 'w') as f:
            f.write(info)

    def persistMemoryCsv(info, output, file):
        header = ['config', 'videoName', 'frame', 'qFrames', 'videoWidth', 'videoHeight',  'Correlation', 'Chi-Square',
                  'Intersection', 'Bhattacharyya', 'diffTime','yoloTime', 'confidence', 'boxes', 'diffTime','totalTime', 'classes', 'confidenceClasses', 'difThreshold', 'Tracker', 'difMethod']
        with open(output + file, 'a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=header)
            csv_writer.writeheader()

            for data in info:
                csv_writer.writerow(data)

