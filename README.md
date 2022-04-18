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
