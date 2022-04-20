import cv2
from fileEnum import File
from processImage import processImage
from predictor import Predictor

def main():
    imagePath = "images\\9_11 (2).JPG"
    img = cv2.imread(imagePath)

    predictor = Predictor()

    outputs = predictor.predict(img)
    predictor.showPredictImage(img=img,outputs=outputs)
    
    file = File.Image

    if file == File.Image:
        processImage(outputs=outputs,img=img)
    elif file == File.Video:
        print(file)

if __name__ == "__main__":
    main()
