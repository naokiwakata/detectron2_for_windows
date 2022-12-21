from enum import Enum
import cv2
import sys
from domain.leaf_predictor import LeafPredictor
from domain.disease_predictor_3class import Disease3ClassPredictor
import numpy as np 

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
    diseasePredictor = Disease3ClassPredictor()
    # 動画を読み込む
    cap = cv2.VideoCapture("video//IMG_6825.MOV")
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    # 保存用の動画のプロパティを指定
    fourcc = cv2.VideoWriter_fourcc(*'XVID')# mp4形式を指定
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('video//new.mp4', fourcc, fps, size,isColor=True)

    # trackerとleafStateを管理する箱
    trackerAndLeafStates = {}
    i = 0 # フレーム数

    while True:
        # フレームを読み込む
        ok, original_frame = cap.read()
        show_frame = np.copy(original_frame)
        
        # フレームが有効であるか確認
        if not ok:
            print('not ok')
            break

        if i%7 == 0: # あるフレームごとに葉っぱを検出しなおす
            # 最初のイニシャライズもここでやる（i == 0）
            outputs = leafPredictor.predict(img=original_frame) #葉を検出
            bounding_boxes = outputs["instances"].pred_boxes.tensor.tolist()
            bounding_boxes = boxesForTracking(bounding_boxes)
            trackerAndLeafStates.clear()

            # BoundingBoxごとにTrackerをイニシャライズ
            for box in bounding_boxes:
                tracker = cv2.legacy.TrackerMedianFlow_create()
                ok = tracker.init(original_frame,box)

                x1 = int(box[0])
                y1 = int(box[1])
                x2 = x1 + int(box[2])
                y2 = y1 + int(box[3])
                
                cutOutImg = np.zeros((y2-y1, x2-x1, 3), np.uint8)
                # 葉1枚ずつ病気か健康を判別させる
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        cutOutImg[y-y1,x-x1]=original_frame[y,x]

                # 病気or健康の予測
                predict = diseasePredictor.predict(img=cutOutImg)
                max_index = np.argmax(predict[0])
                leafState = None
                if max_index == 0: # 病害初期
                    leafState = LeafState.EARLYDISEASE
                elif max_index == 1: # 健康
                    leafState = LeafState.HEALTH
                elif max_index == 2: # 病害後期
                    leafState = LeafState.LATERDISEASE    
                trackerAndLeafStates[tracker] = leafState
            print('detect leaf')
      
        for tracker, leafState in trackerAndLeafStates.items():
                # Update the tracker
                ok, bbox = tracker.update(original_frame)

                # 短径描画
                if ok:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    color = leafState.value
                    # 画像の書き込みはshow_frameに行う
                    cv2.rectangle(show_frame, p1, p2, color, 2, 1)            
        # 表示
        show_frame = cv2.resize(show_frame, (int(width/3), int(height/3)))
        cv2.imshow('load',show_frame)

        # 変換されたフレームを保存
        out.write(original_frame)

        # Qキーでストップ
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        i = i + 1
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('Finish')
    print(i)

class LeafState(Enum):
    HEALTH =  (0,255,0) #Green
    EARLYDISEASE = (0,255,255) #Yellow
    LATERDISEASE = (0,0,255) #Red