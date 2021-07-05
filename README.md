# Object-Detection-YOLO

- YOLO(You Only Look Once) のexample プログラム

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

## リポジトリ構成

```
.
├── README.md
└── notebooks
```

## 環境詳細

- Google Colab
