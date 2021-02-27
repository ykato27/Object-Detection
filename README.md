# YOLOv3
* YOLOv3のテストプログラム

## リポジトリ構成
```
.
└── README.md                 READMEファイル
```

## 環境構築

* condaで仮想環境を構築
```
conda create -n yolov3
```

* condaの仮想環境を有効化
```
conda activate yolov3
```

* ライブラリのインストール
```
conda install tensorflow==1.14.0 keras==2.2.4 pillow matplotlib
```

* YOLOv3をgit cloneする
```
git clone https://github.com/qqwweee/keras-yolo3.git
```

* 学習済みモデルをダウンロード
```
git clone https://github.com/qqwweee/keras-yolo3.git
```

* 学習済みモデルをKeras用に変換
```
wget https://pjreddie.com/media/files/yolov3.weights
```

## 実行手順

* YOLO-v3を実行
```
python yolo_video.py --image
```

* ファイル名を入力
```
Input image filename:＜ファイル名＞
```

## 動作環境
マシンスペック（Mac)
- MacBook Air (Retina, 13-inch, 2018)
- 1.6 GHz デュアルコアIntel Core i5
- 8 GB 2133 MHz LPDDR3
