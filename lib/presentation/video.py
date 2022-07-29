import cv2
import sys
from domain.predictor import Predictor


def loadVideo():
    predictor = Predictor()
    cap = cv2.VideoCapture('leaf_sample2.mov')
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    delay = 1
    window_name = 'frame'
    if not cap.isOpened():
        sys.exit()

    while True:
        ret, frame = cap.read()
        if ret:
            outputs = predictor.predict(img=frame)
            print(outputs)

            img = predictor.getPredictImage(img=frame, outputs=outputs)
            # size 調整
            frame = cv2.resize(cv2.cvtColor(
                img, cv2.COLOR_BGR2RGB), (int(width/3), int(height/3)))
            cv2.imshow(window_name, frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cv2.destroyWindow(window_name)
