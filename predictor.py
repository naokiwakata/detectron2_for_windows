import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

class Predictor:
    _metadata = None
    _predictor = None
    _outputs = None

    def __init__(self):
        # データセットを登録
        register_coco_instances(
            "leaf", {}, "PumpkinLeaf\PumpkinLeaf.json", "PumpkinLeaf/")
        self._metadata = MetadataCatalog.get("leaf")
        setup_logger()

        # 設定を決める
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation\\mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1クラスのみ

        cfg.MODEL.WEIGHTS = os.path.join(
            cfg.OUTPUT_DIR, "C:\\Users\\wakanao\\detectron2-windows\\model_final.pth")  # 絶対パスでなければならないっぽい
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        cfg.MODEL.DEVICE = "cpu"

        # 予測器を作成
        self._predictor = DefaultPredictor(cfg)
    
    def predict(self,img):
        self._outputs = self._predictor(img)
        return self._outputs
    
    def metadata(self):
        return self._metadata

    def showPredictImage(self,img,outputs):
        v = Visualizer(img[:, :, ::-1],
                   metadata=self._metadata,
                   scale=1.0
                   )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('image', v.get_image()[:, :, ::-1])
        cv2.waitKey(0)
