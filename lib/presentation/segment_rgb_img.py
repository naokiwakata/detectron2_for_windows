import cv2
import numpy as np
from domain.leaf_predictor import LeafPredictor
from domain.disease_predictor import DiseasePredictor
from domain.disease_predictor_3class import Disease3ClassPredictor
### RGB画像から葉を検出
### 1枚1枚の病気健康を判別したい

def segment_rgb_img():
    # 葉っぱ検出：インスタンスセグメンテーション
    leafPredictor = LeafPredictor()
    diseasePredictor = DiseasePredictor()
    img_path = "D:\\wakata_research\\2022\\RGBs\\36-40\\IMG_1537.JPG"
    original_img = cv2.imread(img_path)  #判別に回す画像
    draw_img = cv2.imread(img_path) #描画する画像

    leaf_outputs = leafPredictor.predict(img=original_img)
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
        
        cutOutImg = np.zeros((y2-y1+10, x2-x1+10, 3), np.uint8)
        # 葉1枚ずつ病気か健康を判別させる
        for y in range(y1, y2):
            for x in range(x1, x2):
                isLeaf = pred_mask[y, x]
                if(isLeaf):
                    cutOutImg[y-y1,x-x1]=original_img[y,x]

        # 病気or健康の予測
        predict = diseasePredictor.predict(img=cutOutImg)

        # 画像に書き込み
        color = None
        if predict > 0.5: # health
            color = (0, 255, 0) #Green
        else:             # disease
            color = (0, 0, 255) #Red
        # 短径描画
        cv2.rectangle(draw_img, (x1,y1), (x2, y2), color, thickness=5)
    
    # Resize
    height = int(draw_img.shape[0]/5)
    width = int(draw_img.shape[1]/5)
    draw_img = cv2.resize(draw_img, (width, height))
    # Show Image
    cv2.imshow('image', draw_img)
    cv2.waitKey(0)      


### 背景なし＆3クラス
def classify_three_class():
    # 葉っぱ検出：インスタンスセグメンテーション
    leafPredictor = LeafPredictor()
    diseasePredictor = Disease3ClassPredictor()
    img_path = "D:\\wakata_research\\2022\\RGBs\\36-40\\IMG_1537.JPG"
    original_img = cv2.imread(img_path)  #判別に回す画像
    draw_img = cv2.imread(img_path) #描画する画像

    leaf_outputs = leafPredictor.predict(img=original_img)
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
        
        cutOutImg = np.zeros((y2-y1+10, x2-x1+10, 3), np.uint8)
        # 葉1枚ずつ病気か健康を判別させる
        for y in range(y1, y2):
            for x in range(x1, x2):
                cutOutImg[y-y1,x-x1]=original_img[y,x]

        # 病気or健康の予測
        predict = diseasePredictor.predict(img=cutOutImg)
        print(predict)

        # 0：病害初期
        # 1：健康
        # 2：病害後期
        max_value = max(predict[0])
        max_index = np.argmax(predict[0])
        color = None
        if max_index == 0: # 病害初期
            print("EarlyDisease")
            color = (0, 255, 255) #Yellow
        elif max_index == 1: # 健康
            print("Health")
            color = (0, 255, 0) #Green
        elif max_index == 2: # 病害後期
            print("LaterDisease")
            color = (0, 0, 255) #Red
        # 短径描画
        cv2.rectangle(draw_img, (x1,y1), (x2, y2), color, thickness=5)
    
    # Resize
    height = int(draw_img.shape[0]/5)
    width = int(draw_img.shape[1]/5)
    draw_img = cv2.resize(draw_img, (width, height))
    # Show Image
    cv2.imshow('image', draw_img)
    cv2.waitKey(0)      

