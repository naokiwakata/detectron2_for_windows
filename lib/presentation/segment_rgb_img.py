import cv2
import numpy as np
import glob
from domain.leaf_predictor import LeafPredictor
import os

def segment_rgb_img():
    # 葉っぱ検出：インスタンスセグメンテーション
    leafPredictor = LeafPredictor()
    img_path = "D:\\wakata_research\\2022\\RGBs\\36-40\\IMG_1537.JPG"
    img = cv2.imread(img_path) 

    leaf_outputs = leafPredictor.predict(img=img)
    #box内の背景を落とす
    # Prepare
    fields = leaf_outputs['instances'].get_fields()
    pred_boxes = fields['pred_boxes']
    np_boxes = pred_boxes.tensor.to('cpu').detach().numpy().astype(np.int32)
    pred_masks = fields['pred_masks'].to('cpu').detach().numpy()

    for i, box in enumerate(np_boxes):
        pred_mask = pred_masks[i]
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        # Drop Img Not Leaf
        for y in range(y1, y2):
            for x in range(x1, x2):
                isLeaf = pred_mask[y, x]
                if(isLeaf != True):
                    img[y, x] = (0,0,0) # 背景を黒く落とす
        # box内の画像を機械学習にかけて判別させる

        # Draw Predicted Rectrangle
        cv2.rectangle(img, (x1,y1), (x2, y2), (255, 0, 0), thickness=5)
    # Resize
    height = int(img.shape[0]/5)
    width = int(img.shape[1]/5)
    img = cv2.resize(img, (width, height))
    # Show Image
    cv2.imshow('image', img)
    cv2.waitKey(0)    

