# Object-Detection

- Object-Detection のexample プログラム

## リポジトリ構成
```
.
├── README.md
├── config
│   └── NanoDet
│       ├── nanodet-m-original.yml
│       └── nanodet-m.yml
├── data
│   └── NanoDet
│       ├── Annotations
│       ├── ImageSets
│       │   └── Main
│       │       ├── Fish_train.txt
│       │       └── Fish_val.txt
│       ├── JPEGImages
│       └── pascal_label_map.pbtxt
├── docs
│   ├── Date.md
│   └── README.md
├── models
│   └── NanoDet
│       └── workspace
│           └── nanodet_m
│               ├── NanoDet
│               │   └── 2021-11-26-07-32-21
│               │       └── checkpoints
│               │           └── epoch=279-step=279.ckpt
│               ├── logs-2021-11-26-07-32-21
│               │   ├── Train_loss_loss_bbox_Train
│               │   │   └── events.out.tfevents.1637911958.732dd71f9eb4.171.3
│               │   ├── Train_loss_loss_dfl_Train
│               │   │   └── events.out.tfevents.1637911958.732dd71f9eb4.171.4
│               │   ├── Train_loss_loss_qfl_Train
│               │   │   └── events.out.tfevents.1637911958.732dd71f9eb4.171.2
│               │   ├── Train_loss_lr_Train
│               │   │   └── events.out.tfevents.1637911956.732dd71f9eb4.171.1
│               │   ├── Val_metrics_AP_50_Val
│               │   │   └── events.out.tfevents.1637911972.732dd71f9eb4.171.6
│               │   ├── Val_metrics_AP_75_Val
│               │   │   └── events.out.tfevents.1637911972.732dd71f9eb4.171.7
│               │   ├── Val_metrics_AP_l_Val
│               │   │   └── events.out.tfevents.1637911972.732dd71f9eb4.171.10
│               │   ├── Val_metrics_AP_m_Val
│               │   │   └── events.out.tfevents.1637911972.732dd71f9eb4.171.9
│               │   ├── Val_metrics_AP_small_Val
│               │   │   └── events.out.tfevents.1637911972.732dd71f9eb4.171.8
│               │   ├── Val_metrics_mAP_Val
│               │   │   └── events.out.tfevents.1637911972.732dd71f9eb4.171.5
│               │   ├── events.out.tfevents.1637911956.732dd71f9eb4.171.0
│               │   ├── logs.txt
│               │   └── train_cfg.yml
│               ├── logs.txt
│               ├── model_best
│               │   ├── eval_results.txt
│               │   └── model_best.ckpt
│               ├── model_last.ckpt
│               ├── nanodet.onnx
│               └── results0.json
├── notebooks
│   ├── CenterNet
│   │   ├── CenterNet_ObjectsAsPoints.ipynb
│   │   ├── CenterNet_ObjectsAsPoints_3D.ipynb
│   │   └── CenterTrack_Tracking_Objects_as_Points.ipynb
│   ├── DETR
│   │   ├── DETR_demo.ipynb
│   │   ├── DETR_hands_on.ipynb
│   │   └── DETR_panoptic.ipynb
│   ├── Detectron2
│   │   └── Detectron2_Tutorial.ipynb
│   ├── EfficientDet
│   │   └── EfficientDet_Tutorial.ipynb
│   ├── NanoDet
│   │   └── NanoDet_GoogleColab_Training_Sample.ipynb
│   ├── RCNN
│   │   ├── 3_10_MaskRCNN.ipynb
│   │   ├── Detectron_MaskRCNN.ipynb
│   │   ├── FasterRCNN+InceptionResNet_and_ssd+mobilenet.ipynb
│   │   ├── Mask_R_CNN_Demo.ipynb
│   │   ├── Matterport_Mask_RCNN.ipynb
│   │   └── Torchvision_Mask_RCNN.ipynb
│   ├── SSD
│   │   ├── 07_single_shot_multibox_detector.ipynb
│   │   ├── FasterRCNN+InceptionResNet_and_ssd+mobilenet.ipynb
│   │   └── SSD_nvidia_deeplearningexamples.ipynb
│   ├── TensorFlow_Hub_Object_Detection
│   │   ├── Object_Detection_Inference_on_TF_2_and_TF_Hub.ipynb
│   │   ├── Object_detection_tf_hub.ipynb
│   │   └── objdetect_tdmodels.ipynb
│   ├── Traffic_counting_with_OpenCV.ipynb
│   ├── YOLACT
│   │   └── YOLACT_Eval.ipynb
│   └── YOLO
│       ├── YOLOv3_Tutorial.ipynb
│       ├── YOLOv4.ipynb
│       ├── YOLOv4_Darknet_Train_2.ipynb
│       ├── YOLOv4_DeepSort.ipynb
│       ├── YOLOv4_Scaled_Train.ipynb
│       ├── YOLOv4_Training_Tutorial.ipynb
│       ├── YOLOv4_Tutorial.ipynb
│       ├── YOLOv5.ipynb
│       ├── YOLOv5_DeepSort_Pytorch_tutorial.ipynb
│       ├── YOLOv5_Tutorial.ipynb
│       ├── YOLOv5_Workers_example.ipynb
│       ├── YOLOv5_pytorch_train.ipynb
│       └── YOLOv5_tutorials_roboflow.ipynb
├── pyproject.toml
├── requirements.txt
├── setup.cfg
├── src
│   ├── __init__.py
│   └── voc2coco
│       └── voc2coco.py
├── tests
│   └── __init__.py
└── work
```

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

## NanoDet とは

- NanoDetは、超高速で軽量のモバイルアンカーフリーターゲット検出モデル

## YOLACT とは

- YOLACTはインスタンスセグメンテーションアルゴリズムの一種です。

## DETR（DEtection Transformer）

- Transformerを物体検出にはじめて取り入れた手法
- 2020年５月にFacebookから発表
- Encoder-Decoder型Transformerを使った初めての物体検出モデル
- End-to-endなモデルを実現し、人手によるハイパーパラメータの設計を大幅に削減することに成功

## Display notebooks

- [View Jupyter notebooks in nbviewer](https://nbviewer.jupyter.org/github/ykato27/Object-Detection/tree/main/notebooks/)

## 環境詳細

- Google Colab
