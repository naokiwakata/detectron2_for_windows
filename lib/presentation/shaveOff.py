import cv2
import numpy as np


def shaveOff(outputs, img):

    fields = outputs['instances'].get_fields()
    pred_boxes = fields['pred_boxes']
    scores = fields['scores'].to('cpu').detach().numpy()
    pred_classes = fields['pred_classes'].to('cpu').detach().numpy()
    pred_masks = fields['pred_masks'].to('cpu').detach().numpy()
    image_size = outputs['instances'].image_size
    height = image_size[0]
    width = image_size[1]
    base_img = np.zeros((height, width, 3), np.uint8)
    np_boxes = pred_boxes.tensor.to('cpu').detach().numpy()
    for i, box in enumerate(np_boxes):
        pred_mask = pred_masks[i]
        x1 = box[0].astype(int)
        y1 = box[1].astype(int)
        x2 = box[2].astype(int)
        y2 = box[3].astype(int)
        for y in range(y1, y2):
            for x in range(x1, x2):
                isLeaf = pred_mask[y, x]
                if(isLeaf == True):
                    base_img[y, x] = img[y, x]
    cv2.imshow('only leaf', base_img)
    cv2.waitKey(0)
