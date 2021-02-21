# KoikatuGen

このプログラムは[コイカツ](http://www.illusion.jp/preview/koikatu/)のキャラクターデータを学習し、キャラクターのランダム生成を可能とすることを目的としています。
現状のところ、VariationalAutoEncoderで生成を行っています。
また、学習対象とするデータは[コイカツ公式アップローダー](http://up.illusion.jp/koikatu_upload/chara/)にあるMod無しのデータのみとしています(学習の安定性のため)。

## 生成データの例
![](https://i.imgur.com/rubgXw7.png)
40epochの学習が終わった状態でのモデルで生成しています。
他の例は[生成データ一覧](#生成データ一覧)にあります。

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

## 生成データ一覧
- 10epoch
![](https://i.imgur.com/WvMYDCN.png)
- 20epoch
![](https://i.imgur.com/vOVWeli.png)
- 30epoch
![](https://i.imgur.com/HZ5GsZO.png)
- 35epoch
![](https://i.imgur.com/iLa6gIs.png)
- 40epoch
![](https://i.imgur.com/rubgXw7.png)
- 45epoch
![](https://i.imgur.com/uLB4qzR.png)
- 50epoch
![](https://i.imgur.com/bYMnMfh.png)

見れば分かるように、45epoch及び50epochの生成データではほとんど同じような顔が生成されています。おそらく過学習の問題だと考えられます。

## 使用モジュール
- [KoikatuCharaLoader](https://github.com/great-majority/KoikatuCharaLoader)
- [KoikatuWebAPI](https://github.com/great-majority/KoikatuWebAPI)

を使用しています。