# Axell AI Contest 2025 - サンプルコード

Axell AI Contest 2025向けに、[Ultralytics](https://github.com/ultralytics/ultralytics) を用いて YOLO11 物体検出モデルを学習するサンプルです。

## サンプルコード構成

* [`TRAIN.md`](./TRAIN.md): 本資料
* [`config.py`](./config.py): Python スクリプト版学習サンプル([`train.py`](./train.py))向けの設定ファイル
* [`prepare_dataset.py`](./prepare_dataset.py): Python スクリプト版学習サンプル([`train.py`](./train.py))向けのデータセット前処理ツール (Ultralytics向け)
* [`requirements.txt`](./requirements.txt): 依存パッケージ情報
* [`train.ipynb`](./train.ipynb): Notebook版学習サンプル
* [`train_colab.ipynb`](./train_colab.ipynb): Google Colaboratory版学習サンプル
* [`train.py`](./train.py): Python スクリプト版学習サンプル
* [`submit`](./submit): SIGNATE投稿用サンプルファイル一式

## データセットについて

コンテストページで配布される `README.md` を参照してください。  

なお、Python版およびNotebook版学習サンプルでは、サンプルスクリプトと同じ場所に、配布されているデータセット `dataset.zip` を展開したものとして説明をしております。

展開例
```bash
├── TRAIN.md
├── ...
├── train.py
└── dataset/
    ├── annotations/
    │   └── train.json
    └── images/
        ├── T1.jpg
        ├── T2.jpg
        └── ...
```

## 環境構築

### 手元環境での学習(Pythonスクリプト版/Notebook版)

ここでは配布しているサンプルコード一式を事前に `path/to/sample_code` へ展開したものとします。

#### Conda(推奨)

手元環境にcondaをインストールして環境を構築する手順です。  
なお、NVIDIA製GPUを用いて学習を行なう場合、事前に最新のGPUドライバーの導入が必要です。  

1. [Miniconda](https://docs.anaconda.com/miniconda/install/) のセットアップ手順に従い、condaをインストール

2. 端末(Mac/Linux)もしくはAnaconda Prompt(Windows)を起動

3. AI コンテスト向けに仮想環境を作成
```bash
conda create -n ai_challenge python=3.12
```

4. 仮想環境を有効化
```bash
conda activate ai_challenge
```

5. PyTorchをインストール
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

CUDAのバージョンを変更する場合は `pytorch-cuda` の後のバージョンを変更してください。
また、手動でインストールする場合は [venv](#venv) での構築手順を参考に torch をインストールしてください。

6. その他のパッケージを導入
```bash
cd path/to/sample_code
pip install -r requirements.txt
```

メモ: 有効化した仮想環境から抜ける場合は、次のコマンドを実行します
```bash
conda deactivate
```

#### venv

手元のPythonの実行環境を用いて学習環境を構築することも可能です。  
なお、NVIDIA製GPUを用いて学習を行なう場合、事前に最新のGPUドライバー/CUDAの導入が必要です。  
インストールするCUDAのバージョンは pytorch が対応しているもの (2025年7月現在 11.8,12.6,12.8) を導入してください。

1. AI コンテスト向けに仮想環境を作成
```bash
python3 -m venv ai_challenge
```

2. 仮想環境を有効化
例 linux:
```bash
source ai_challenge/bin/activate
```
例 windows:
```bash
ai_challenge/Scripts/Activate.ps1
```

3. PyTorchをインストール
環境によりインストール方法が異なりますので、[公式ドキュメント](https://pytorch.org/get-started/locally/)を参考にインストールします。  

例 CUDA12.8の場合  
https://pytorch.org/get-started/locally/ 上で
* PyTorch Build = Stable
* Your OS = お使いのOS
* Package = pip
* Compute Platform = CUDA12.8
を選択し、 `Run this command` の内容を実行します。

4. その他のパッケージを導入
```bash
pip install -r requirements.txt
```

メモ: 有効化した仮想環境から抜ける場合は、次のコマンドを実行します
```bash
deactivate
```

#### Notebookで学習する場合

Notebookで学習をする場合、condaもしくはvenvで環境構築をしたあと、追加で依存ライブラリの導入が必要です。
```shell
pip install notebook ipywidgets
```


### Google Colaboratory

Googleのアカウントをお持ちで無い場合は事前にアカウントの作成を行なってください。

## 学習

### 手元環境での学習(Pythonスクリプト版)

1. 仮想環境を有効化します

例 condaの場合
```bash
conda activate ai_challenge
```

例 venvの場合
```bash
source ai_challenge/bin/activate
```

2. COCO形式のデータセットをUltralytics向けに変換し、学習用データセットを学習用と評価用へ分離します

```bash
python prepare_dataset.py \
    --src <PATH_TO_EXTRACTED_DATASET> \
    --dst <PATH_TO_OUTPUT_DATASET>
```

**NOTE**
このスクリプトはCOCO形式のデータセットに学習用(train.json)、評価用(val.json)、テスト用(test.json)のそれぞれに対してUltralytics向けへ変換を行います。  
評価用データセットが存在しない場合は変換後に学習用データセットを学習用と評価用へ分離します。  
なお、学習用データと評価用データ分離の比率は `config.py` の `TRAIN_SPLIT` で制御が可能です。

例:
```bash
python prepare_dataset.py \
    --src ./dataset \
    --dst ./dataset_yolo
```

`<PATH_TO_OUTPUT_DATASET>/data.yaml` にYOLO11の学習に必要な設定ファイルが生成されます。

3. 学習を実行します (YOLO11 nanoの例)
```bash
python train.py \
    -c <PATH_TO_OUTPUT_DATASET>/data.yaml \
    -m yolo11n \
    -o <PATH_TO_OUPTUT_DIRECTORY>
```

例
```bash
python train.py \
    -c ./dataset_yolo/data.yaml \
    -m yolo11n \
    -o ./output
```

学習が完了すると `<PATH_TO_OUTPUT_DIRECTORY>/train_YYYY-MM-DD_HH-MM-SS` フォルダーへ学習結果が生成されます。  
また、学習により最も良い精度となったモデルファイルは `<PATH_TO_OUPTUT_DIRECTORY>/train_YYYY-MM-DD_HH-MM-SS/weights/best.pt` として保存されます。
また、このモデルは学習終了時に 評価用データセットを用いて自動的に評価が行なわれます。

### 手元環境での学習(Notebook版)

1. 仮想環境を有効化します

例 condaの場合
```bash
conda activate ai_challenge
```

例 venvの場合
```bash
source ai_challenge/bin/activate
```

2. train.ipynbが配置されている場所でnotebookを起動します

```shell
python -m notebook
```

**NOTE 1**
Visual Studio Code を用いて編集・実行も可能です。

**NOTE 2**
デフォルトのポートを用いてWebサーバーが立ち上がります。  
起動するポートを変更したい場合は `--port <PORT_NUMBER>` オプションを与えてください。  
また、他のマシンから Notebook を操作したい場合は、　`--ip=*` オプションを与えてください。

3. NotebookをWebブラウザから開きます
Notebook 起動時にターミナルに出力されたアドレスとトークンを用いてアクセスしてください。

例:  `http://localhost:8888/tree?token=XXXXXXXXXXXXXX`

4. サンプル Notebook `train.ipynb` を開きます

5. 全てのセルを上から順に実行します

6. 最後のセルまで到達すると学習済みモデル (`outputs/train_YYYY-MM-DD_HH-MM-SS/weights/best.pt`) がサンプルコード (`train.ipynb`) と同じ場所へ生成されます


### Using Google Colaboratory

1. Google Driveを開き、作業用フォルダー `AI_Contest` を作成します

なお、作業用フォルダーを別の名前で作成する場合は、サンプルコードの `DRIVE_DIR` を変更してください。


2. `AI_Contest` フォルダーにデータセットのzipファイルを配置します

配置後のファイル構成は次のようになります。
```
AI_Contest
└── dataset.zip
```

3. [Google Colaboratory](https://colab.research.google.com) を開きます

4. `アップロード` から `参照` を選択し、サンプルのノートブック [train_colab.ipynb](./train_colab.ipynb) を選択し、アップロードします。

5. `ランタイム` から `ランタイムのタイプを変更` を選択し、 ランタイムのタイプ を `Python 3` に、ハードウェアアクセラレーター は利用したいものを選択し、保存を選択します

6. `全てのセルを実行` を選択し、セルを順に実行します
実行時にGoogleDriveへのアクセス許可画面が表示されますので、ログインと全てのパーミッションへ許可を与えてください。

6. 最後のセルまで到達すると学習済みモデル (`outputs/train_YYYY-MM-DD_HH-MM-SS/weights/best.pt`) がGoogle Driveの作業用フォルダーへ保存されます

## 投稿

学習により得られたモデルファイル (`best.pt`) をサンプル投稿ファイル一式の `model` フォルダーの下にコピーします。
最後に `submit` フォルダーごとzipファイルで圧縮し、zipファイルを投稿します。

最終的な投稿用zipファイルの内容は次の形式となります。

```
submit
├── model
│   └── best.pt
├── src
│   └── predictor.py
└── requirements.txt
```

