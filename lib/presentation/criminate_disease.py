import cv2
from nnabla.utils import nnp_graph
from nnabla.utils.image_utils import imread

### 病気判別のデモ

def criminate_disease():
    # 学習済みニューラルネットワークの読み込み
    model_path = "model\disease_crimination.nnp"
    nnp = nnp_graph.NnpLoader(model_path)
    # 推論用ニューラルネットワークの取得
    graph = nnp.get_network('MainRuntime',batch_size=1)

    # 入力変数xの取得
    x = list(graph.inputs.values())[0]
    # 出力変数yの取得
    y = list(graph.outputs.values())[0]

    # xに画像を代入
    img_path = "images\disease\913 (1)_1.png"
    img = imread(img_path)
    img = cv2.resize(img, dsize=(128, 128))
    x.d = img.reshape(1,3,128,128) * 1.0/255

    # 推論の実行
    y.forward()
    print(y.d)

