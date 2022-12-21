import cv2
import sys
from domain.leaf_predictor import LeafPredictor

### 動画を読み込むやつ

## Trackingで使いやすいように中身を整形
def boxesForTracking(boxes):
    bboxes = []
    for box in boxes:
        bbox = [box[0],box[1],box[2]-box[0],box[3]-box[1]]
        bboxes.append(bbox)
    return bboxes

# 単一の葉をトラッキング
def trackBox():
    leafPredictor = LeafPredictor()
    video_path = 'video\IMG_6825.MOV'
    
    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    if not cap.isOpened():
        sys.exit()

    tracker = cv2.legacy.TrackerMedianFlow_create()

    # Read the first frame
    ok, frame = cap.read()
    outputs = leafPredictor.predict(img=frame)
    first_img = leafPredictor.getPredictedImg(img=frame,outputs=outputs)

    bounding_boxes = outputs["instances"].pred_boxes.tensor.tolist()
    bounding_boxes = boxesForTracking(bounding_boxes)
    first_bbox = bounding_boxes[0]
    ok = tracker.init(frame, first_bbox)

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

        # Display the frame
        frame = cv2.resize(frame, (int(width/3), int(height/3)))
        cv2.imshow("Tracking", frame)
        
        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 複数の葉をトラッキング
def trackBoxes():
    leafPredictor = LeafPredictor()
    # 動画を読み込む
    cap = cv2.VideoCapture("video//IMG_6825.MOV")

    # 保存用の動画のプロパティを指定
    fourcc = cv2.VideoWriter_fourcc(*'XVID')# mp4形式を指定
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('video//new.mp4', fourcc, fps, size,isColor=True)

    trackers = [] #複数の葉をトラッキングするための箱
    i = 0 # フレーム数

    while True:
        # フレームを読み込む
        ok, frame = cap.read()
        
        # フレームが有効であるか確認
        if not ok:
            print('not ok')
            break

        if i%7 == 0: # あるフレームごとに葉っぱを検出しなおす
            # 最初のイニシャライズもここでやる（i == 0）
            outputs = leafPredictor.predict(img=frame) #葉を検出
            bounding_boxes = outputs["instances"].pred_boxes.tensor.tolist()
            bounding_boxes = boxesForTracking(bounding_boxes)
            trackers.clear() # Trackerをリセットする（検出精度の向上も兼ねる）

            # BoundingBoxごとにTrackerをイニシャライズ
            for box in bounding_boxes:
                tracker = cv2.legacy.TrackerMedianFlow_create()
                ok = tracker.init(frame,box)
                trackers.append(tracker)  
            print('detect leaf')
      
        # 検出した葉ごとに短径を描画
        for tracker in trackers:
                # Update the tracker
                ok, bbox = tracker.update(frame)
                # 短径描画
                if ok:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

        # 表示
        cv2.imshow('load',frame)

        # 変換されたフレームを保存
        out.write(frame)

        # Qキーでストップ
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        i = i + 1
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('Finish')
    print(i)