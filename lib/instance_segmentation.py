from dataclasses import field
import cv2
from fileEnum import File
from processImage import processImage
from predictor import Predictor
import numpy as np
from shaveOff import shaveOff
import glob

def instanceSegmentation():
    imagePath = "images\9_23 (3).JPG"
    img = cv2.imread(imagePath)  # <class 'numpy.ndarray'>

    predictor = Predictor()

    outputs = predictor.predict(img=img)

    # jpgs = glob.glob('testDataLeaf\\*.jpg')
    # for imagePath in jpgs:
    #     img = cv2.imread(imagePath)  # <class 'numpy.ndarray'>
    #     outputs = predictor.predict(img=img)
    #     fields = outputs['instances'].get_fields()
    #     pred_boxes = fields['pred_boxes']
    #     print(len(pred_boxes))
    #     predictor.showPredictImage(img=img, outputs=outputs)



    shaveOff(outputs=outputs,img=img) #葉っぱのみを切り抜く

    fields = outputs['instances'].get_fields()
    pred_boxes = fields['pred_boxes']
    scores = fields['scores'].to('cpu').detach().numpy()
    pred_classes = fields['pred_classes'].to('cpu').detach().numpy()
    pred_masks = fields['pred_masks'].to('cpu').detach().numpy()

    image_size = outputs['instances'].image_size
    height = image_size[0]
    width = image_size[1]

    predictor.showPredictImage(img=img, outputs=outputs)
