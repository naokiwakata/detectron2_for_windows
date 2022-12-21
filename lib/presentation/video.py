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

def saveVideo():
    # 動画を読み込む
    cap = cv2.VideoCapture("video//IMG_6825.MOV")

    # 動画のパラメータを指定する
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # mp4形式を指定
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 出力用の動画を作成する
    out = cv2.VideoWriter('video//new.mp4', fourcc, fps, size)

    # 動画をフレームごとに処理する
    while True:
        # フレームを読み込む
        ok, frame = cap.read()
        
        # フレームが有効であるか確認
        if not ok:
            break
        
        # フレームをグレースケールに変換する
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray',frame)
        
        # 変換されたフレームを保存する
        out.write(gray)

    # 出力用の動画をリリースする
    out.release()
    cap.release()
    print('finish')

def loadVideo():
    print(cv2)
    # 動画を読み込む
    cap = cv2.VideoCapture("video//IMG_6825.MOV")

    # 動画のパラメータを指定する
    fourcc = cv2.VideoWriter_fourcc(*'XVID')# mp4形式を指定
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 出力用の動画を作成する
    # isColor=Falseと定義しなければ上手く保存されない（https://plugout.hateblo.jp/entry/2020/01/10/085552）
    out = cv2.VideoWriter('video//new.mp4', fourcc, fps, size,isColor=False)

    # 動画をフレームごとに処理する
    while True:
        # フレームを読み込む
        ok, frame = cap.read()
        
        # フレームが有効であるか確認
        if not ok:
            print('not ok')
            break
        
        # フレームをグレースケールに変換する
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('load',gray)
        # 変換されたフレームを保存する
        out.write(gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('Finish')