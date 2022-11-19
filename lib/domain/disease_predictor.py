from nnabla.utils import nnp_graph
import cv2


class DiseasePredictor:
    _predictor = None
    _x = None
    _y = None

    def __init__(self):
        # モデル読み込み
        model_path = "model\disease_crimination.nnp"
        nnp = nnp_graph.NnpLoader(model_path)

        # 推論用ニューラルネットワークを取得
        graph = nnp.get_network('MainRuntime', batch_size=1)

        # 入力変数の取得
        self._x = list(graph.inputs.values())[0]
        self._y = list(graph.outputs.values())[0]

    def predict(self, img):
        img = cv2.resize(img, dsize=(128, 128))
        img = img.reshape(1, 3, 128, 128)
        img = img * 1.0/255
        self._x.d = img

        # 推論実行
        self._y.forward()
        return self._y.d
