from dataclasses import field
import cv2
from domain.leaf_predictor import LeafPredictor
from domain.disease_predictor import DiseasePredictor
from presentation.shaveOff import shaveOff
from presentation.crip import cripBackground


def instanceSegmentation():
    # 画像読み込み
    imagePath = "images\9_23 (3).JPG"
    #imagePath = "images\910 (1)_23.png"
    img = cv2.imread(imagePath)  # <class 'numpy.ndarray'>

    # 病気判別：CNN
    diseasePredictor = DiseasePredictor()

    # 葉っぱ検出：インスタンスセグメンテーション
    leafPredictor = LeafPredictor()
    leaf_outputs = leafPredictor.predict(img=img)

    criped_imgs = cripBackground(outputs=leaf_outputs, img=img)
    print(f"検出枚数： {len(criped_imgs)}")
    for criped_img in criped_imgs:
        # 一枚ごとに病気を判定
        result = diseasePredictor.predict(img=criped_img)
        print(result)
        cv2.imshow('only leaf', criped_img)
        cv2.waitKey(0)

    # 判別精度を高める
    # 画像にラベル付けして表示したい

    # jpgs = glob.glob('testDataLeaf\\*.jpg')
    # for imagePath in jpgs:
    #     img = cv2.imread(imagePath)  # <class 'numpy.ndarray'>
    #     outputs = predictor.predict(img=img)
    #     fields = outputs['instances'].get_fields()
    #     pred_boxes = fields['pred_boxes']
    #     print(len(pred_boxes))
    #     predictor.showPredictImage(img=img, outputs=outputs)

    shaveOff(outputs=leaf_outputs, img=img)  # 葉っぱのみを切り抜く

    fields = leaf_outputs['instances'].get_fields()
    pred_boxes = fields['pred_boxes']
    scores = fields['scores'].to('cpu').detach().numpy()
    pred_classes = fields['pred_classes'].to('cpu').detach().numpy()
    pred_masks = fields['pred_masks'].to('cpu').detach().numpy()

    image_size = leaf_outputs['instances'].image_size
    height = image_size[0]
    width = image_size[1]

    leafPredictor.showPredictImage(img=img, outputs=leaf_outputs)
