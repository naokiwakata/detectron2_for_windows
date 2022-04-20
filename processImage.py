import cv2

def processImage(outputs,img):
    boxes = outputs["instances"].pred_boxes
    tensor = boxes.tensor
    number_box = boxes.tensor.shape[0]

    for i in range(number_box):
        box = tensor[i]
        x1 = round(box[0].item())-10  # 画像の表示領域を10px広くする
        y1 = round(box[1].item())-10
        x2 = round(box[2].item())+10
        y2 = round(box[3].item())+10
        cut_img = img[y1:y2, x1:x2]
        cv2.imshow('cut_image', cut_img)
        cv2.waitKey(0)