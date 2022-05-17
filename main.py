from dataclasses import field
import cv2
from fileEnum import File
from processImage import processImage
from predictor import Predictor
import numpy as np


def main():
    imagePath = "images\\9_11 (2).JPG"
    img = cv2.imread(imagePath)  # <class 'numpy.ndarray'>

    predictor = Predictor()

    outputs = predictor.predict(img=img)

    fields = outputs['instances'].get_fields()
    pred_boxes = fields['pred_boxes']
    scores = fields['scores'].to('cpu').detach().numpy()
    pred_classes = fields['pred_classes'].to('cpu').detach().numpy()
    pred_masks = fields['pred_masks'].to('cpu').detach().numpy()

    image_size = outputs['instances'].image_size
    height = image_size[0]
    width = image_size[1]

    # 葉っぱ以外の部分を落とす
    base_img = np.zeros((height,width,3),np.uint8)
    np_boxes = pred_boxes.tensor.to('cpu').detach().numpy()
    for i, box in enumerate(np_boxes):
        pred_mask = pred_masks[i]
        x1 = box[0].astype(int)
        y1 = box[1].astype(int)
        x2 = box[2].astype(int)
        y2 = box[3].astype(int)

        for y in range(y1, y2):
            for x in range(x1, x2):
                isLeaf = pred_mask[y,x]
                if(isLeaf == True):
                    base_img[y,x] = img[y, x]

    cv2.imshow('only leaf', base_img)
    cv2.waitKey(0)

    predictor.showPredictImage(img=img, outputs=outputs)

    file = File.Image
    #file = File.Video

    if file == File.Image:
        #processImage(outputs=outputs, img=img)
        print(file)
    elif file == File.Video:
        print(file)


if __name__ == "__main__":
    main()
