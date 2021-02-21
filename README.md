# KoikatuGen

このプログラムは[コイカツ](http://www.illusion.jp/preview/koikatu/)のキャラクターデータを学習し、キャラクターのランダム生成を可能とすることを目的としています。
現状のところ、VariationalAutoEncoderで生成を行っています。
また、学習対象とするデータは[コイカツ公式アップローダー](http://up.illusion.jp/koikatu_upload/chara/)にあるMod無しのデータのみとしています(学習の安定性のため)。

## 実行準備
```
$ git clone https://github.com/tropical-362827/KoikatuGen
$ git submodule update --init --recursive
$ pip install -r ./requirements.txt
```

## create_dataset.py

`kk_charas`フォルダ内のキャラクターデータを順番に開き、データのベクトル化を行います。
ベクトル化したデータは`kk_charas.csv`に出力されます。
公式アップローダーから削除されているかどうかの判定を行うため、公式アップローダーの情報を取得する処理を含みます。
さらに、png画像の名前は全て公式アップローダーで振られているidの数字でなければなりません。(この辺の処理は面倒なので、2021/02/21時点でベクトル化したデータを`kk_charas.rar`に入れてあります。解凍して使ってください。)

## vae_train.py

`kk_charas.csv` のデータを学習し、学習モデルを5epoch毎に`vae_models`に出力します。

## vae_generate.py

学習モデルからキャラクターデータを生成します。プログラムのオプションは
```
$ python ./vae_generate.py (vae_models内のフォルダ名:"20210222_0328"など)
```
のように指定します。
服やキャラクター情報などは`default.png`のものが使用されます。(デフォルトではちかりんです😄)