import argparse
import cv2
import imutils
import sys
import timeit
from datetime import datetime
from models.tem import tem
from models.codelets.compareImages import compareHist
from models.pam import yolo
from models.codelets.trackingObjects import *

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=False,
                help='path to input video/image')
ap.add_argument('-c', '--config', required=False,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=False,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=False,
                help='path to text file containing class names')
args = ap.parse_args()

def main():
    gwt(0.7, 0.16, 'Bhattacharyya', "csrt")

    #tests()
    #confidenceRange()
def gwt(confValue, difT, difM, dMet):
    video   = '/Users/lukercio/Downloads/videoobjecttracking/TrackingDataset/BoBot/Vid_C_juice/img%04d.jpg'

    config  = '/Users/lukercio/PycharmProjects/lida/data/yolov3.cfg'
    weights = '/Users/lukercio/PycharmProjects/weights/yolov3.weights'
    trained = '/Users/lukercio/PycharmProjects/lida/data/yolov3.txt'
    output  = '/Users/lukercio/Documents/mestrado/results/'
    fileOutputName = str(confValue) + datetime.now().strftime("-%d-%m-%Y-%H-%M-%S") + '.csv'
    COPARISON_ON = False
    DIF_THRESHOLD = difT
    TRACKER_METHOD = dMet
    DIF_METHOD = difM


    # Starting measure time
    startTotal = timeit.default_timer()

    # Open video
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    lframe = frame

    j = 0
    tempMemory = tem()

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Count video frames: ', length)

    height, width, layers = frame.shape
    print(height)
    print(width)
    print(layers)

    new_h =180
    new_w = 240

    # Initialize dictionary to be persisted
    temDict = [{'config': config,
                'videoName': video,
                'frame': None,
                'qFrames': length,
                'videoWidth': new_w,
                'videoHeight': new_h,
                'Correlation': None,
                'Chi-Square': None,
                'Intersection': None,
                'Bhattacharyya': None,
                'diffTime': None,
                'yoloTime': None,
                'confidence': None,
                'boxes': None,
                'totalTime': None,
                'classes': None,
                'confidenceClasses': None,
                'difThreshold': DIF_THRESHOLD,
                'Tracker': TRACKER_METHOD,
                'difMethod': DIF_METHOD} for k in range(length)]
    trackers = cv2.MultiTracker_create()
    validate = True

    while (ret and j < length):
        # save in dictionary current frame
        temDict[j]['frame'] = j

        frameR = cv2.resize(frame, (int(new_w), int(new_h)))

        if (COPARISON_ON):
            try:
                startDiff = timeit.default_timer()

                m3, m2, m1, m0 = compareHist(frameR, lframe)

                stopDiff = timeit.default_timer()
                print('Time Diff (sec): ', (stopDiff - startDiff))
                temDict[j]['diffTime'] = stopDiff - startDiff

                # Save comparision to memory
                temDict[j]['Correlation'] = m0
                temDict[j]['Chi-Square'] = m1
                temDict[j]['Intersection'] = m2
                temDict[j]['Bhattacharyya'] = m3

                dif = temDict[j][DIF_METHOD]

            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise
        else:
            # make diff very large to force yolo usage
            dif = 999999999

        print('Diff between images: ', dif)
        if (dif > DIF_THRESHOLD or j == 0):
            start = timeit.default_timer()
            tempMemory = yolo(frameR, config, weights, trained, confValue)
            stop = timeit.default_timer()
            print('Time Yolo (sec): ', stop - start)
            temDict[j]['yoloTime'] = stop - start

            print('Verifying if image is not empty')
            if (tempMemory is not None):
                lframe = frameR
                temDict[j]['boxes'] = tempMemory.boxes
                temDict[j]['classes'] = [tempMemory.classes[element] for element in tempMemory.class_ids]
                temDict[j]['confidenceClasses'] = tempMemory.confidences

                print('Image is not empty. Assigning it to memory')
                EmptyImage = False
            else:
                EmptyImage = True

            validate = True
        else:
            print('tempMemory: ', tempMemory)
            if tempMemory is not None and len(tempMemory.indices) is not 0:

                print('BOXES --------> ', tempMemory.boxes)

                lframe = imutils.resize(lframe, width=600)

                tr = TRACKER_METHOD
                validate, success, boxes, trackers = multObjectFocus(frame, trackers, tr, tempMemory.boxes, validate)
                print('Boxes tracking: ', boxes)
                print('SUCCESS: ', success)
                tempMemory.set_boxes(boxes)

                if(len(boxes) == 0):
                    print('Image is empty.')
                    EmptyImage = True
                else:
                    EmptyImage = False

        # loop over the bounding boxes and draw then on the frame
        if (not EmptyImage):
            bcount = 0
            for box in tempMemory.boxes:

                label = str(tempMemory.classes[tempMemory.class_ids[tempMemory.indices[bcount][0]]])
                color = tempMemory.COLOR[tempMemory.class_ids[tempMemory.indices[bcount][0]]]

                (x, y, w, h) = [int(v) for v in box]

                #print('Drawing boundbox!! ',(x, y, w, h))

                cv2.rectangle(frameR, (x, y), (x + w, y + h), color, 2)

                cv2.putText(frameR, label, (int(x - 10), int(y - 10)), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color, 1)
                print('Drawing boundbox!! ',(x, y, w, h))

                bcount = bcount + 1


                print('Current Frame: ', j)

        print('Loop Finished')
        cv2.imshow("pam", frameR)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        j += 1
        print('\n\n')
        ret, frame = cap.read()
        print('End of video: ', not ret)
        print('-------------------------------------------------------------------------')

    cap.release()
    cv2.destroyAllWindows()

    stopTotal = timeit.default_timer()
    print('Time Total (sec): ', (stopTotal - startTotal))
    temDict[j-1]['totalTime'] = stopTotal - startTotal

    tem.persistMemoryCsv(temDict, output, fileOutputName)



def confidenceRange():
    range = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    print('Starting test with different confideces thresholds of yolo')
    for r in range:
        gwt(r)

    print('End of test - confidenceRange')
    exit(0)


def tests():
    rangeT = [0.97, 5, 19, 0.16]

    methods = ['Correlation', 'Chi-Square', 'Intersection', 'Bhattacharyya']


    detcMeth = ["csrt", "kcf", "boosting","mil", "tld", "medianflow", "mosse"]


    print('Starting test with different thresholds of compare attention codelet')

    for d in detcMeth:
        for i in range(0,4):
            gwt(0.7, rangeT[i], methods[i], d)

    print('End of test - confidenceRange')
    exit(0)
if __name__== "__main__":
  main()

