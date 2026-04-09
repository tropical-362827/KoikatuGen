# KoikatuGen

このプログラムは[コイカツ](http://www.illusion.jp/preview/koikatu/)のキャラクターデータを学習し、キャラクターのランダム生成を可能とすることを目的としています。

現状のところ、

- Variational Autoencoder
- CTGAN

で生成を行っています。

↓このプログラムでやっていることの解説をQiitaにて行っています！

[深層学習モデル(VAE)で"コイカツ!"のキャラクターを生成する](https://qiita.com/tropical-362827/items/e9d12f54ad0dda5d2e30)

## ゲーム中から呼び出すMOD
このプログラムで得られた学習モデルを使ってゲーム中からキャラクターを生成するMOD、[KoikatuGen-Plugin](https://github.com/tropical-362827/KoikatuGen-Plugin)を作りました。
学習パラメータを使用する方法は[KoikatuGen-PluginのREADME.md](https://github.com/tropical-362827/KoikatuGen-Plugin#koikatugen%E3%81%A7%E5%AD%A6%E7%BF%92%E3%81%97%E3%81%9F%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E3%82%92%E4%BD%BF%E3%81%86--using-parameters-trained-by-koikatugen)に記載してあります。

## 生成データの例

### Variational Autoencoder
![](https://i.imgur.com/vxxCdqI.png)
90epochの学習が終わった状態でのモデルで生成しています。
他の例は[生成データ一覧](#生成データ一覧)にあります。

## プロジェクト構造

```
KoikatuGen/
├── data/
│   ├── raw/                  # キャラクターPNGファイルを置くディレクトリ
│   ├── preprocessed/
│   │   └── kk_charas.parquet # 前処理済みデータ
│   └── templates/
│       └── default.png       # 生成時のテンプレートキャラ
├── koikatugen/
│   ├── dataset/
│   │   ├── loader.py         # データセット作成
│   │   ├── transforms.py     # キャラデータのベクトル化
│   │   └── schema.py         # カラム定義
│   ├── models/
│   │   └── vae.py            # VAEモデル
│   └── scripts/
│       ├── create_dataset.py # PNGキャラデータからデータセット作成
│       ├── train_vae.py      # VAEで学習する
│       ├── generate_vae.py   # VAEで生成
│       ├── train_ctgan.py    # CTGANで学習する
│       └── generate_ctgan.py # CTGANで生成
└── outputs/
    └── vae/
        └── {timestamp}/     # チェックポイント・生成結果
    └── ctgan/
        └── {timestamp}/     # チェックポイント・生成結果
```

## セットアップ

*Python 3.12以上* と [uv](https://github.com/astral-sh/uv) が必要です。

```
$ git clone https://github.com/tropical-362827/KoikatuGen
$ make init
```

## Makefileの使い方

### 1. データの用意

#### 前処理済みデータのダウンロード

[Hugging Face](https://huggingface.co/datasets/tropical-362827/KoikatuGen) から前処理済みデータを取得します。

```
$ make pull-preprocessed
```

#### 生のキャラデータからデータセット作成(Optional)

`data/raw/kk_chara/` にキャラクターPNGを配置した上で実行します。
ベクトル化したデータを `data/preprocessed/kk_charas.parquet` に出力します。
スキップされたファイルの理由は `kk_charas_skipped.json` に記録されます。

```
$ make create_dataset
```

あるいは、別の場所にあるデータから作成:

```
$ uv run python -m koikatugen.scripts.create_dataset \
            --chara-dir /path/to/chara-data \
            --out-parquet /path/to/preprocessed.parquet
```

### 2. 学習

デフォルトでは `kk_charas.parquet` を使って学習します。チェックポイントは `outputs/{model}/{timestamp}/` に5epoch毎に保存されます。

```
$ make train_vae
$ make train_ctgan
```

あるいは、自分で作成したデータを学習:
```
$ uv run python -m koikatugen.scripts.train_vae \
            --parquet /path/to/preprocessed.parquet
```

#### CTGANで学習

`kk_charas.parquet` を使ってCTGANを学習します。チェックポイントは 5 epoch ごとに `outputs/ctgan/{timestamp}/epoch_XXX.pkl`、最終モデルは `outputs/ctgan/{timestamp}/model.pkl` に保存されます。

```
$ make train_ctgan
```

### 3. 生成

学習済みチェックポイントからキャラクターを生成します。

```
$ make generate_vae CHECKPOINT=outputs/vae/{timestamp}/epoch_100.pth
$ make generate_ctgan CHECKPOINT=outputs/ctgan/{timestamp}/model.pkl
```

生成されたPNGは `outputs/{model}/{timestamp}/generated/` に出力されます。
服やキャラクター情報は `data/templates/default.png` のものが使用されます。

生成されたPNGは `outputs/ctgan/{timestamp}/generated/` に出力されます。

## 生成データ一覧

### Variational Autoencoder
<details>
<summary>開く</summary>

- 10epoch
![](https://i.imgur.com/cqoVZBf.png)
- 20epoch
![](https://i.imgur.com/JZTIaF2.png)
- 30epoch
![](https://i.imgur.com/v15ZKoA.png)
- 40epoch
![](https://i.imgur.com/oG7VA1R.png)
- 50epoch
![](https://i.imgur.com/y1CMgPO.png)
- 60epoch
![](https://i.imgur.com/z2G5VMp.png)
- 70epoch
![](https://i.imgur.com/U9duOlA.png)
- 80epoch
![](https://i.imgur.com/rSNlv5g.png)
- 90epoch
![](https://i.imgur.com/vxxCdqI.png)
- 100epoch
![](https://i.imgur.com/i47jVO9.png)

</details>

## 使用ライブラリ

- [kkloader](https://github.com/great-majority/KoikatuCharaLoader)
