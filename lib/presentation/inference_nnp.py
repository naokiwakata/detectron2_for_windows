from nnabla.utils import nnp_graph
from nnabla.utils.image_utils import imread
import cv2

def inference():
    # モデル読み込み
    nnp = nnp_graph.NnpLoader('model\\results.nnp')

    # 推論用ニューラルネットワークを取得
    graph = nnp.get_network('MainRuntime',batch_size=1)

    # 入力変数の取得
    x = list(graph.inputs.values())[0]
    y = list(graph.outputs.values())[0]

    # 入力変数に画像を代入
    path = 'images\9_17 (12).JPG'
    img = cv2.imread(path)
    img = cv2.resize(img, dsize=(128, 128))
    img = img.reshape(1,3,128,128)
    img = img * 1.0/255
    x.d = img

    # 推論実行
    y.forward()
    print(y.d)