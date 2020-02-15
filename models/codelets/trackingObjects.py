import cv2

def trackerFocus(boundbox, frame, lframe, i, tracker):
    w = int(boundbox[i][2])
    h = int(boundbox[i][3])
    img_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gray_lframe = cv2.cvtColor(lframe, cv2.COLOR_BGR2GRAY)

    bbox = (boundbox[i][0], boundbox[i][1], (boundbox[i][0]+w), (boundbox[i][1]+h))


    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int( bbox[2]), int( bbox[3]))



    print('Start tracking')
    print('before track bbox: ', bbox)

    ok = tracker.init(lframe, bbox)
    print('Status init ok: ', ok)

    ok, bbox = tracker.update(frame)
    print('bbox found: ', type(bbox[0]))
    print('bbox[0]: ', bbox[0])
    print('OK: ', ok)
    print('boundbox: ', boundbox)
    boundbox = (int(bbox[0]), int(bbox[1]),w,h)
    return boundbox, ok

def trackerFocus2(boundbox, frame, lframe, i, tracker):
    y = int(boundbox[i][1])
    x = int(boundbox[i][0])
    w = int(boundbox[i][2])
    h = int(boundbox[i][3])

    bb_y = int(y*0.8)
    bb_x = int(x*0.8)
    bb_w = int(w*2.5)
    bb_h = int(h*1.2)

    print('bb_y: ', bb_y)
    print('bb_x: ', bb_x)
    print('bb_w: ', bb_w)
    print('bb_h: ', bb_h)

    bbx_e_y = y - bb_y
    bbx_e_x = x - bb_x
    bbx_e_w = bb_w - w
    bbx_e_h = bb_h - h

    print('bbx_e_y: ', bbx_e_y)
    print('bbx_e_x: ', bbx_e_x)
    print('bbx_e_w: ', bbx_e_w)
    print('bbx_e_h: ', bbx_e_h)

    bbx_e = (bbx_e_x, bbx_e_y, (bbx_e_x + bbx_e_w), (bbx_e_y + bbx_e_h))
    print('bbx_e: ', bbx_e)


    baseCropFrame = lframe[y:y + h, x:x + w].copy()
    boundboxCropFrame = frame[bb_y:(bb_y + bb_h), bb_x:(bb_x + bb_w)].copy()
    print('imgs cropped')


    bbox = (boundbox[i][0], boundbox[i][1], (boundbox[i][0]+w), (boundbox[i][1]+h))
    print('bbox: ', bbox)

    ok = tracker.init(boundboxCropFrame, bbx_e)
    print('ok: ', ok)

    ok, bbox = tracker.update(baseCropFrame)

    print('bbox found: ', type(bbox[0]))
    print('OK: ', ok)
    print('boundbox: ', boundbox)
    boundbox = (int(bbox[0]), int(bbox[1]),w,h)
    return boundbox, ok


def cetralizeFocus(boundbox, frame, lframe, i):

    print('BOXES: ', boundbox)
    print('i: ', i)
    boxes = []
    y = int(boundbox[i][1])
    x = int(boundbox[i][0])
    w = int(boundbox[i][2])
    h = int(boundbox[i][3])

    print(x,y,w,h)
    baseCropFrame = lframe[y:y + h, x:x + w].copy()

    bb_y = int(y-5)
    bb_x = int(x-5)
    bb_w = int(w*2.5)
    bb_h = int(h*1.2)

    if (bb_x <= 0):
        bb_x = 0
    if (bb_y <=0):
        bb_y = 0
    print('Expanded: ',bb_x,bb_y,bb_w,bb_h)
    boundboxCropFrame = frame[bb_y:(bb_y + bb_h), bb_x:(bb_x + bb_w)].copy()

    img_gray_baseCropFrame = cv2.cvtColor(baseCropFrame, cv2.COLOR_BGR2GRAY)
    img_gray_boundboxCropFrame = cv2.cvtColor(boundboxCropFrame, cv2.COLOR_BGR2GRAY)



    res = cv2.matchTemplate(img_gray_boundboxCropFrame,img_gray_baseCropFrame,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(boundboxCropFrame,top_left, bottom_right, 255, 0)
    print('left: ', top_left,' right: ', bottom_right)


    boxes = [bb_x + top_left[0], bb_y + top_left[1],  w,  h]
    print('soma:', boxes)

    return boxes

    cv2.imshow("baseCropFrame", baseCropFrame)
    cv2.imshow("boundboxCropFrame", boundboxCropFrame)
    cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[0]+w, boxes[1]+h), 255, 2)
    cv2.imshow("New Frame", frame)

    cv2.waitKey(0)
    return boxes

    exit(0)

def multObjectFocus(frame, trackers, tr, bbxs, validate):
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    print('Initial bbxs: ', bbxs)
    print('Len initial bbxs: ', len(bbxs))
    print('Validate: ', validate)

    print('Before trackers.getObjects(): ', trackers.getObjects())
    if (validate):
        resetTr = True
    else:
        resetTr = False
    for box in bbxs:
        if(validate):
            if(resetTr):
                trackers = cv2.MultiTracker_create()
            box = (box[0], box[1], box[2], box[3])
            tracker = OPENCV_OBJECT_TRACKERS[tr]()
            trackers.add(tracker, frame, box)
            resetTr = False
        print('Box add:', box)
    validate = False

    print('After trackers.getObjects(): ', trackers.getObjects())
    (success, boxes) = trackers.update(frame)
    print('New bbxs: ', boxes)
    print('Len new bbxs: ', len(boxes))


    return validate, success, boxes, trackers