import cv2
import numpy as np
import glob
from domain.leaf_predictor import LeafPredictor
import os

def crip():
    # 葉っぱ検出：インスタンスセグメンテーション
    leafPredictor = LeafPredictor()
    path = "C:\\Users\\wakanao\\Desktop\\dataset\\*\\disease\\*.png"
    img_paths = glob.glob(path)

    save_folder = "C:\\Users\\wakanao\\Desktop\\dataset\\disease_no_background\\"
    #save_folder = "C:\\Users\\wakanao\\Desktop\\dataset\\health_no_background\\"
    for img_path in img_paths:
        filename = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path) 
        leaf_outputs = leafPredictor.predict(img=img)
        leafPredictor.showPredictImage(img=img,outputs=leaf_outputs)
        criped_imgs = cripBackground(outputs=leaf_outputs,img=img)

        for index, criped_img in enumerate(criped_imgs):
            cv2.imwrite(save_folder + filename + "_" +str(index) + ".png", criped_img)


def cripBackground(outputs, img):

    fields = outputs['instances'].get_fields()
    pred_boxes = fields['pred_boxes']
    scores = fields['scores'].to('cpu').detach().numpy()
    pred_classes = fields['pred_classes'].to('cpu').detach().numpy()
    pred_masks = fields['pred_masks'].to('cpu').detach().numpy()
    image_size = outputs['instances'].image_size
    height = image_size[0]
    width = image_size[1]
    np_boxes = pred_boxes.tensor.to('cpu').detach().numpy()
    criped_imgs = []
    # 見つけたBoxの数だけFor文回す
    for i, box in enumerate(np_boxes):
        pred_mask = pred_masks[i]
        x1 = box[0].astype(int)
        y1 = box[1].astype(int)
        x2 = box[2].astype(int)
        y2 = box[3].astype(int)
        black_img = np.zeros((y2-y1+20, x2-x1+20, 3), np.uint8)
        for y in range(y1, y2):
            for x in range(x1, x2):
                isLeaf = pred_mask[y, x]
                if(isLeaf == True):
                    black_img[y-y1+10, x-x2-10] = img[y, x]
        criped_imgs.append(black_img)
    
    return criped_imgs
        