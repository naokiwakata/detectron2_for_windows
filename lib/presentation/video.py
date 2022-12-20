import cv2
import sys
from domain.leaf_predictor import LeafPredictor

### 動画を読み込むやつ

def loadVideo():
    leafPredictor = LeafPredictor()
    video_path = 'video\IMG_6825.MOV'
    save_path = 'C://Users//wakanao//Desktop//rec.mp4'
    
    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1
    window_name = 'frame'
    
    fmt =cv2.VideoWriter_fourcc(*'mp4v')
    rec = cv2.VideoWriter(save_path,
                        fmt,
                        fps, (int(width), int(height)))
    if not cap.isOpened():
        sys.exit()


    while True:
        ret, frame = cap.read()

        if ret:
            outputs = leafPredictor.predict(img=frame)
            # print(outputs)
            # 検出されたBounding Boxを保存
            bounding_boxes = outputs["instances"].pred_boxes.tensor.tolist()

            img = leafPredictor.getPredictedImg(img=frame, outputs=outputs)
            # size 調整
            #frame = cv2.resize(cv2.cvtColor(
            #    frame, cv2.COLOR_RGB2BGR), (int(width/3), int(height/3)))
            frame = cv2.resize(img, (int(width/3), int(height/3)))
            cv2.imshow(window_name, frame)
            rec.write(frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cv2.destroyWindow(window_name)


def boxesForTracking(boxes):
    bboxes = []
    for box in boxes:
        bbox = [box[0],box[1],box[2]-box[0],box[3]-box[1]]
        bboxes.append(bbox)
    return bboxes


def trackBox():
    leafPredictor = LeafPredictor()
    video_path = 'video\IMG_6825.MOV'
    
    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1
    window_name = 'frame'

    if not cap.isOpened():
        sys.exit()

    tracker = cv2.legacy.TrackerMedianFlow_create()
    tracker2 = cv2.legacy.TrackerMedianFlow_create()
    tracker3 = cv2.legacy.TrackerMedianFlow_create()

    # Read the first frame
    ok, frame = cap.read()
    outputs = leafPredictor.predict(img=frame)
    first_img = leafPredictor.getPredictedImg(img=frame,outputs=outputs)

    bounding_boxes = outputs["instances"].pred_boxes.tensor.tolist()
    bounding_boxes = boxesForTracking(bounding_boxes)
    first_bbox = bounding_boxes[0]
    ok = tracker.init(frame, first_bbox)

    second_bbox = bounding_boxes[1]
    ok = tracker2.init(frame, second_bbox)

    second_bbox = bounding_boxes[2]
    ok = tracker3.init(frame, second_bbox)

    cv2.imshow('first_img', first_img)
    cv2.waitKey(0)   

    while True:
        # Read a frame from the video stream
        ok, frame = cap.read()
        
        # Check if the frame is valid
        if not ok:
            break
        
        # Update the tracker
        ok, bbox = tracker.update(frame)
        # Draw the bounding box on the frame
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

        ok, bbox = tracker2.update(frame)       
        # Draw the bounding box on the frame
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        ok, bbox = tracker3.update(frame)       
        # Draw the bounding box on the frame
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        # Display the frame
        frame = cv2.resize(frame, (int(width/3), int(height/3)))
        cv2.imshow("Tracking", frame)
        
        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def boxesForTracking(boxes):
    bboxes = []
    for box in boxes:
        bbox = [box[0],box[1],box[2]-box[0],box[3]-box[1]]
        bboxes.append(bbox)
    return bboxes

def trackBoxes():
    leafPredictor = LeafPredictor()
    video_path = 'video\IMG_6825.MOV'
    
    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    if not cap.isOpened():
        sys.exit()

    ok, frame = cap.read()
    # 複数のBoundingBoxに対するTrackerの箱を用意
    trackers = []
    i = 0 
    while True:
        # Read a frame from the video stream
        ok, frame = cap.read()
        
        # Check if the frame is valid
        if not ok:
            break
        
        if i%15 == 0: # あるフレームごとに葉っぱを検出しなおす
            # 最初のイニシャライズもここでやる（i == 0）
            outputs = leafPredictor.predict(img=frame)
            bounding_boxes = outputs["instances"].pred_boxes.tensor.tolist()
            bounding_boxes = boxesForTracking(bounding_boxes)
            trackers.clear()
            for box in bounding_boxes:
                tracker = cv2.legacy.TrackerMedianFlow_create()
                ok = tracker.init(frame,box)
                trackers.append(tracker)  
            print('re:detectReaf')
      
        for tracker in trackers:
                # Update the tracker
                ok, bbox = tracker.update(frame)
                # Draw the bounding box on the frame
                if ok:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

        # Display the frame
        frame = cv2.resize(frame, (int(width/3), int(height/3)))
        cv2.imshow("Tracking", frame)
        
        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i = i+1