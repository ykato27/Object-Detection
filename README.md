# Object-Detection

- Object-Detection のexample プログラム

## YOLO とは

- YOLO のバージョンはV.2からV.5まで存在(2021/7 現在)
- YOLO v.3はDarknet というC 言語コードで書かれたスクリプトをコンパイルした実行ファイルを用いて処理を実施
  - [詳細HP](https://pjreddie.com/)
- Aleksey Bochkovskiy氏がDarknet のGithub に依拠して、YOLOV.4を開発
  - [Github](github.com/AlexeyAB/darknet)
- Roboflow という会社がPytorch 向けのYOLOV4 を発表
  - [Github](https://github.com/roboflow-ai/pytorch-YOLOv4)
- 2020年6月にUltralytics 社がPytorch 向けのYOLOv.5 を公開
  - [Github](https://github.com/ultralytics/yolov5)
  - YOLOv5 は、検出精度と演算負荷に応じてs、m、l、xまでの4モデル存在
  - YOLOv5 のsモデルを使用することで、YOLOv3 のfullモデルに近い性能を、1/4以下の演算量で実現
- 2020年12月にYOLOv4 並びにEfficientDet の性能を超えると言われる Scaled-YOLOv4 モデルが公開

## SSD とは

- SSD モデルでは、デフォルト・ボックスという概念を使用します。
  - デフォルト・ボックスとは画像マップの各セルを中心とする４種類または６種類の（各辺の長さが異なる）四角形のことです。
  - SSD　では、これらのうちサイズ38x38 、 19x19 、 10x10 、 5x5 、3x3 、 1x1 の特徴マップにおいて、各マップをデフォルト・ボックスで覆います。 
  - 従って、総数で8732個（38x4+19x6+10x6+5x6+３x４＋1x4)）のデフォルト・ボックスが存在します。
  - これらのボックスの中に何らかの物体が存在するか否かを調べて、その物体を識別します。
  - 物体識別の確率が最も高いデフォルト・ボックスだけを残し、それを物体検出のバウンディング・ボックスとします。

## Detectron2 とは

- GPUでのモデル学習を前提としない物体検出のライブラリの一つ
- Facebook　がオープンソース・ソフトウエア型式で公開
  - [Github](https://github.com/facebookresearch/detectron2)

## Faster R-CNN とは

- 2015年にMicrosoft が発明した物体検出アルゴリズムです。

## リポジトリ構成

```
.
├── README.md
└── notebooks
```

## 環境詳細

- Google Colab
