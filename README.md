# Detectron2をローカルで動かす（植物の検出）
<img src="https://user-images.githubusercontent.com/65523426/163788316-8fbcf0de-49df-472c-853c-faedb6d83151.png" width="500">

## Detectron2とは
https://github.com/facebookresearch/detectron2

Detectron2とはFacebook AIが開発した、PyTorchベースの物体検出のライブラリです。 様々なモデルとそのPre-Trainedモデルが実装されており、下記のように、Bounding boxやInstance Segmentation等の物体検出を簡単に実装することができます。

##### 参考記事
- 全体の流れ  
https://qiita.com/bear_montblanc/items/5bb1ad3506718120682d

- coco-anotatorの使い方  
https://qiita.com/PoodleMaster/items/39830656d69d34a39f34

- Dockerインストール  
https://docs.docker.jp/docker-for-windows/install.html  
https://chigusa-web.com/blog/windows%E3%81%ABdocker%E3%82%92%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB%E3%81%97%E3%81%A6python%E7%92%B0%E5%A2%83%E3%82%92%E6%A7%8B%E7%AF%89/  
https://qiita.com/PoodleMaster/items/75edc1744b0a4986c1c8  
https://zenn.dev/kathmandu/articles/4a86c3d75b93c3

- GoogleColabratory  
[公式](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)  
[自作](https://colab.research.google.com/drive/1XLbOV9x-MQo__WdDnQLxxN0IGJqx4lsE?hl=ja#scrollTo=eeK_hvuzlPtV)

## 環境構築 
https://medium.com/@yogeshkumarpilli/how-to-install-detectron2-on-windows-10-or-11-2021-aug-with-the-latest-build-v0-5-c7333909676f

こちらの記事を参考にWindowsでdetectron2を動かす
- Anacondaをインストール  
https://www.python.jp/install/anaconda/windows/install.html  
condaコマンドが使えることを確認
```
conda info
```
- 記事のCUDAインストールはスキップ（GPUを使用しないため)  

- conda環境を作る
```
conda create -n detectron2(名前はなんでもok) python=3.7
```

- conda環境に移動
```
conda activate detectron2
```

- PyTorchをインストール(バージョンは変えない方が良いかも。以下のバージョンを僕は入れました)
```
pip install torch==1.10.0
```
```
pip install torchaudio==0.10.0
```
```
pip install torchvision==0.11.1
```
- Microsoft Visual Studioを最新にする？（記事要参照）  
C++関連でエラーが出た気がする。それが最新にしたことで解消された  
https://self-development.info/%E3%80%8Cmicrosoft-visual-c-14-0-or-greater-is-required-%E3%80%8D%E3%81%8C%E5%87%BA%E3%81%9F%E5%A0%B4%E5%90%88%E3%81%AE%E5%AF%BE%E5%87%A6%E6%96%B9%E6%B3%95/

- CythonとPycocotoolsをインストール
```
pip install cython
```
```
pip install “git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

- OpenCVをインストール
```
pip install opencv-python
```

detectron2_for_windowsをclone
```
git clone https://github.com/naokiwakata/detectron2_for_local.git
```
作業フォルダをdetectron2-windowsに移動
```
cd detectron2-windows
```
detectron2をclone
```
git clone https://github.com/facebookresearch/detectron2.git
```
もしくは自分のリポジトリにforkしてきたdetectron2をclone
```
git clone https://github.com/naokiwakata/detectron2.git
```
```
python -m pip install -e detectron2
```

これで動かせるはず！！！！

