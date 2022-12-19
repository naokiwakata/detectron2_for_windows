from presentation.instance_segmentation import instanceSegmentation
from presentation.video import loadVideo
from presentation.crip import cripBackground,crip
from presentation.segment_rgb_img import segment_rgb_img,classify_three_class
from presentation.criminate_disease import criminate_disease
import cv2

def main():
    classify_three_class()
    segment_rgb_img()
    loadVideo()

if __name__ == "__main__":
    main()
