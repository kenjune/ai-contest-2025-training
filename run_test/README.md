# 配布データと応募用ファイル作成方法の説明

本コンペティションで配布されるデータと応募用ファイルの作成方法や投稿する際の注意点について説明する.

1. [配布データ](#配布データ)
1. [応募用ファイルの作成方法](#応募用ファイルの作成方法)
1. [投稿時の注意点](#投稿時の注意点)

## 配布データ

配布されるデータは以下の通り.

- [配布データと応募用ファイル作成方法の説明](#配布データと応募用ファイル作成方法の説明)
  - [配布データ](#配布データ)
    - [README](#readme)
    - [配布データ(学習用データ)](#配布データ学習用データ)
    - [動作確認用のプログラム](#動作確認用のプログラム)
    - [応募用サンプルファイル](#応募用サンプルファイル)
  - [応募用ファイルの作成方法](#応募用ファイルの作成方法)
    - [ディレクトリ構造](#ディレクトリ構造)
    - [環境構築](#環境構築)
    - [predictor.pyの実装方法](#predictorpyの実装方法)
      - [Predictor](#predictor)
        - [get\_model](#get_model)
        - [predict](#predict)
    - [推論テスト](#推論テスト)
      - [検証データの作成](#検証データの作成)
      - [推論の実行](#推論の実行)
  - [投稿時の注意点](#投稿時の注意点)

### README

本ファイル(README.md)で, 配布用データの説明と応募用ファイルの作成方法を説明したドキュメント. マークダウン形式で, プレビューモードで見ることを推奨する.

### 配布データ(学習用データ)

ファイルはdataset.zipであり, 解凍すると以下のようなディレクトリ構造となるデータが生成される.

```bash
dataset
├── annotations
│   └── train.json     学習用データ
├── images
│   ├── T1.jpg
│   ├── T2.jpg
│   └── ...
└── ...
```

`jpg`は画像データで`json`は対応するアノテーションデータである. 画像データはボトルや缶が撮影された写真の想定である. アノテーションデータは以下のようなcocoフォーマット形式のjsonファイルである.

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "T1.jpg",
      "height": 1080,
      "width": 1793
    }, ...]
    "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 43,
      "bbox": [
        532,
        367,
        199,
        497
      ]
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 42,
      "bbox": [
        1094,
        457,
        179,
        420
      ]
    },
    ...],
    "categories": [
    {
      "id": 1,
      "supercategory": "can",
      "name": "light_sugar_coffe1_185ml"
    },
    {
      "id": 2,
      "supercategory": "can",
      "name": "sweet_coffe1_180ml"
    },
    ...]
}
```

`images`, `annotations`, `categories`をキーとして、対応するデータが格納されている.

`images`は画像idと画像ファイル名、画像サイズが情報として記載されている.

- `id`: 画像id
- `file_name`: 対応する画像idのファイル名
- `height`: 画像サイズ(縦)
- `width`: 画像サイズ(横)

`annotations`はアノテーション情報が記載されている.

- `id`: アノテーションid
- `image_id`: アノテーションに対応する画像id
- `category_id`: カテゴリid(詳細は`categories`参照)
- `bbox`: 矩形情報(`[左上の座標(x方向), 左上の座標(y方向), width(x軸の幅), height(y軸の高さ)]`). 

`categories`はカテゴリ情報が記載されている.

- `id`: カテゴリid
- `supercategory`: スーパーカテゴリ(缶・ボトルのいずれか)
- `name`: カテゴリ名

このデータセットでは缶とボトルの物体検出を行うため, それ以外の画像が写っていた場合, 該当する物体は検出対象外となる.

### 動作確認用のプログラム

動作確認用プログラム一式はrun_test.zipであり, 解凍すると以下のようなディレクトリ構造のデータが生成される.

```bash
run_test
├── submit              投稿予定のフォルダを置く
│   ├── model              学習済みモデルを置くディレクトリ
│   ├── src                Pythonのプログラムを置くディレクトリ
│   └── requirements.txt   投稿時に追加で必要なライブラリ一覧(投稿時にはこちらの内容が反映される)
├── src                runtimeで動作しているプログラムを格納したディレクトリ
│   ├── generator.py
│   ├── runner.py
│   └── validator.py
├── docker-compose.yml 分析環境を構築するためのdocker-composeファイル
├── Dockerfile         分析環境元のDockerファイル
├── requirements.txt   分析環境作成時に追加で必要となったライブラリを格納(ローカル上の検証で使用する)
└── run.py　　　　　　　実装した推論プログラムを実行するプログラム
```

```bash
run_test
├── submit              投稿予定のフォルダを置く
│   ├── model              学習済みモデルを置くディレクトリ
│   ├── src                Pythonのプログラムを置くディレクトリ
│   └── requirements.txt   投稿時に追加で必要なライブラリ一覧(投稿時にはこちらの内容が反映される)
├── src                runtimeで動作しているプログラムを格納したディレクトリ
│   ├── generator.py
│   ├── runner.py
│   └── validator.py
├── docker-compose.yml 分析環境を構築するためのdocker-composeファイル
├── Dockerfile         分析環境元のDockerファイル
├── requirements.txt   分析環境作成時に追加で必要となったライブラリを格納(ローカル上の検証で使用する)
└── run.py　　　　　　　実装した推論プログラムを実行するプログラム
```
※ **submitフォルダ内のrequirements.txtは, 実際にモデルを投稿する際に使用される**.ローカル上で検証する場合は, run_testフォルダ内のrequirements.txtに必要なライブラリを記載すること.  ローカルでの検証が完了したら, 必要なライブラリをsubmitフォルダのrequirements.txtに記載することを推奨する.

使い方の詳細は[応募用ファイルの作成方法](#応募用ファイルの作成方法)を参照されたい.

### 応募用サンプルファイル

応募用サンプルファイルはsample_submit.zipとして与えられる. 解凍すると以下のようなディレクトリ構造のデータが生成される.

```bash
sample_submit
├── model             学習済みモデルを置くディレクトリ
├── src
│   └── predictor.py  最初のプログラムが呼び出すファイル
└── requirements.txt  追加で必要なライブラリ一覧
```

実際に作成する際に参照されたい.

## 応募用ファイルの作成方法

応募用ファイルは学習済みモデルを含めた, 推論を実行するためのソースコード一式をzipファイルでまとめたものとする.

### ディレクトリ構造

以下のようなディレクトリ構造となっていることを想定する.

```bash
.
├── model              必須: 学習済モデルを置くディレクトリ
│   └── ...
├── src                必須: Pythonのプログラムを置くディレクトリ
│   ├── predictor.py   必須: 最初のプログラムが呼び出すファイル
│   └── ...            その他のファイル (ディレクトリ作成可能)
└── requirements.txt   任意: 追加で必要なライブラリ一覧
```

- 学習済みモデルの格納場所は"model"ディレクトリを想定しする.
  - 学習済みモデルを使用しない場合でも空のディレクトリを作成する必要がある.
  - 名前は必ず"model"とすること.
- Pythonのプログラムの格納場所は"src"ディレクトリを想定している.
  - 学習済みモデル等を読み込んで推論するためのメインのソースコードは"predictor.py"を想定している.
    - ファイル名は必ず"predictor.py"とすること.
  - その他推論を実行するために必要なファイルがあれば作成可能.
  - ディレクトリ名は必ず"src"とすること.
- 実行するために追加で必要なライブラリがあれば, その一覧を"requirements.txt"に記載することで, 評価システム上でも実行可能となる.
  - インストール可能で実行可能かどうか予めローカル環境で試しておくこと.

### 環境構築

評価システムと同じ環境を用意する. Dockerファイルがrun_test.zip内に含まれているので, そちらを実行して環境構築を行うことを推奨する(GPUが搭載されている環境で構築することが望ましい).

イメージ名: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`

Dockerから環境構築する場合, Docker Engineなど, Dockerを使用するために必要なものがない場合はまずはそれらを導入しておく. [Docker Desktop](https://docs.docker.com/get-docker/)を導入すると必要なものがすべてそろうので, 自身の環境に合わせてインストーラをダウンロードして導入しておくことが望ましい. 

現状, Linux, Mac, Windowsに対応している. そして, `/path/to/run_test`に同封してある`docker-compose.yml`で定義されたコンテナを, 以下のコマンドを実行することで立ち上げる.

```bash
$ cd /path/to/run_test
$ docker compose up -d
...
```

`docker-compose.yml`は好きに編集するなりして, 自身が使いやすいように改造してもよい. GPUが使えてCUDAを有効化したい場合は以下のように編集することでコンテナ内で使用することができる.

```yaml
services:
  dev1:
    build:
      context: .
      dockerfile: Dockerfile
    image: signate/runtime-gpu:axell2025_env
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    container_name: axell_2025
    ports:
      - "8080:8080"
    volumes:
      - .:/workspace
    working_dir: /workspace
    tty: true
```

無事にコンテナが走ったら, 必要に応じてデータをコンテナへコピーする.

```bash
$ docker cp /path/to/some/file/or/dir {コンテナ名}: {コンテナ側パス}
... 
```

そして, 以下のコマンドでコンテナの中に入り, 分析や開発を行う.

```bash
$ docker exec -it {コンテナ名} bash
...
```

`コンテナ名`には`docker-compose.yml`の`services`->`dev1`->`container_name`に記載の値を記述する. デフォルトでは`/path/to/run_test`をコンテナ側の`/workspace`へバインドマウントした状態(`/path/to/run_test`でファイルの編集などをしたらコンテナ側にも反映される. 逆もしかり.)で, `/workspace`からスタートする. 追加でPythonライブラリをインストールしたい場合は例えば`requirements.txt`によりコンテナの中でインストール可能.

```bash
# コンテナに入った後
$ pip install -r requirements.txt
...
```

CUDA環境を構築した場合, 実際にCUDAがコンテナ内で有効になっているかどうかは以下のコマンドで確認できる.(対応するライブラリのインストールが必要)

```bash
# コンテナに入った後
$ python -c "import torch; print(torch.cuda.is_available())"
True
$ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
...{GPUデバイスのリスト}...
$ python -c "import paddle; paddle.utils.run_check()"
Running verify PaddlePaddle program ... 
...
PaddlePaddle works well on 1 GPU.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

コンテナ内でCUDAが認識しなかった場合, 配布してあるDockerfileの`nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`をお使いのGPUのcudaのバージョンに合わせるなどの修正が必要となる.

### predictor.pyの実装方法

以下のクラスとメソッドを実装すること.

#### Predictor

推論実行のためのクラス. 以下のメソッドを実装すること.

##### get_model

モデルを取得するメソッド. 以下の条件を満たす必要がある.

- クラスメソッドであること.
- 引数model_path(str型)を指定すること.
  - 学習済みモデルが格納されているディレクトリのパスが渡される.
- 学習済みモデルの読み込みに成功した場合はTrueを返す.
  - モデル自体は任意の名前(例えば"model")で保存しておく.

##### predict

推論を実行するメソッド. 以下の条件を満たす必要がある.

- クラスメソッドであること.
- 引数input(numpy配列)を指定すること.
  - 推論対象となる画像データがnumpy配列で渡される(配色はBGR想定のため予測モデルに応じて変更すること).
  - `get_model`メソッドで読み込んだ学習済みモデルを用いて対応する画像に対する予測を行う想定である.
- 渡された画像ファイルに映る缶やボトルの予測結果を以下のようなdictに格納し, このdictデータをlist内に格納する.(実装例も参照されたい.)
  - 予測するバウンディングボックスの数が**5つ以内**となるようにする（`predict`メソッドが返すリストの長さは**最大5**となる）

`{"category_id": 1, "bbox": [0, 0, width, height], "score": 1.0}`

- この形式はcoco形式に合わせたものだが, image_idとannotation_idについては, run.pyで付与するため不要.

以下は実装例.

```python
import cv2

class Predictor(object):
    @classmethod
    def get_model(cls, model_path):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.

        Returns:
            bool: The return value. True for success, False otherwise.
        """
        # load some model(s)
        cls.model = None
        return True

    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input (numpy.ndarray): Image array loaded by OpenCV in BGR format. Shape should be (height, width, 3).

        Returns:
            List[Dict]: Inference results for a given input. Maximum of 5 dict.
        """
        # get image shape
        height, width, _ = input.shape
        
        predict_list = []
        predict_list.append({"category_id": 1, "bbox": [0, 0, width, height], "score": 1.0})
        return predict_list
```

応募用サンプルファイル"sample_submit.zip"も参照すること.

### 推論テスト

推論を行うプログラムが実装できたら, 正常に動作するか確認する.

以下, コンテナ内での作業とする.

#### 検証データの作成

前述した[動作確認用のプログラム](#動作確認用のプログラム)では検証データを使って動作確認, 精度評価を行う.

本コンペティションでは学習用データセットのみの配布となるため, 別途検証データセットを作成する必要がある.

検証データの作成は, 配布されている`train.json`と同じようなcocoフォーマット形式のjsonファイルを作成すること.

参考として、`make_val.py`には学習用データと検証用データの分割例を記載する.

本コードは以下のコマンドで実行できる.(データセットのパスは適宜修正すること)

```bash
$ python make_val.py  --load-anno-path　./dataset/annotations/train.json　--save-anno-path　./dataset/annotations/custom_val.json --val-ratio 0.25 --seed 42
...
```

各自行うデータセットの分割に合わせて、コードの編集を行うこと.

#### 推論の実行

配布データを用いてモデル学習などを行い, [動作確認用のプログラム](#動作確認用のプログラム)を用いて検証データで, 推論を実行する.

```bash
$ python run.py  --exec-dir ./submit/src --input-data-dir ./dataset --input-name annotations/val.json --tmp-eval-name tmp_val.json --result-dir ./results --result-name result.json
...
```

- `--exec-dir`には実装した予測プログラム("predictor.py")が存在するパス名を指定する.デフォルトは`./submit/src`
- `--input-data-dir`には検証データのディレクトリ名を指定する.デフォルトは`./dataset`
- `--input-name`には検証データのアノテーションファイルを指定する.デフォルトは`annotations/custom_val.json`
- `--tmp-eval-name`には`pycocotools`による検証データを評価するための一時的なアノテーションファイルを指定する.デフォルトは`tmp_val.json`
- `--result-dir`には推論結果の格納先のディレクトリを指定する. デフォルトは`./results`
- `--result-name`には推論結果ファイルの名前を指定する. デフォルトは`result.json`

検証データは各自ファイルを対応する位置に移動させておくこと.

実行に成功すると, 推論時間(prediction_second)などが出力され, `{result_dir}/{result_name}`として推論結果ファイルが保存される.

## 投稿時の注意点

投稿する前に自身のローカル環境などで実行テストを行い, エラーなく実行できるか確認すること. 投稿時にエラーが出た場合, 以下のことについて確認してみる.

- 追加でインストールしたライブラリをrequirements.txtに含めているか
- `predict`メソッドを呼んだときに必ず`list`型を返しているか
- 予測するバウンディングボックスの数が5を超えていないか（`predict`メソッドが返すリストの長さは**最大5**となる）
- 予測結果のデータ型が間違っていないか.
  - `category_id`はint型, `bbox`はlist型, `score`はfloat型とする必要があり, これらを`dict`に格納する必要がある.