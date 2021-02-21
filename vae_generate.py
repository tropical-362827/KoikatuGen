
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from create_dataset import get_dataframe, category_to_onehot, kkchara_to_vector, dataframe_to_kkchara
from tensorflow.python.keras.models import model_from_json, Model
from tensorflow.keras import backend as K
from KoikatuCharaLoader import KoikatuCharaData
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import os
import argparse

latent = 800

parser = argparse.ArgumentParser("KoikatuGen vae_generate.", add_help=False)
parser.add_argument("learned_date", action="store", type=str, help="Specify the folder name where the training model is located under 'vae_models' (ex:20210221_1402)")
args = parser.parse_args()

@tf.function
def sampling(args):
    z_mean, z_sigma = args
    epsilon = K.random_normal(shape=(latent,), mean=0., stddev=1.)
    return z_mean + z_sigma * epsilon

df = pd.read_csv("./kk_charas.csv", index_col=0)
df = category_to_onehot(df)

date = args.learned_date
dirpath = os.path.join("./vae_models", date)
genpath = os.path.join("./vae_generated", date)
if not os.path.exists(genpath):
    os.makedirs(genpath)

for epoch in range(0, 100+1, 5):
    filename = "{:03}_%s.%s".format(epoch)
    filepath = os.path.join(dirpath, filename)

    vae_encoder = model_from_json(open(filepath % ("encoder", "json"), "r").read())
    vae_encoder.load_weights(filepath % ("encoder", "h5"))
    vae_decoder = model_from_json(open(filepath % ("decoder", "json"), "r").read(), custom_objects={"sampling":sampling})
    vae_decoder.load_weights(filepath % ("decoder", "h5"))

    mu = np.random.randn(10, latent)
    sigma = np.exp(np.random.randn(10, latent)*0.5) 
    y = vae_decoder.predict((mu, sigma))

    kc_df = pd.DataFrame(y, columns=df.columns)
    t = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    for i,r in kc_df.iterrows():
        kc = dataframe_to_kkchara(r, KoikatuCharaData.load("./default.png"))
        kc.parameter["lastname"] = "{:03}_{:03}".format(epoch,i)
        kc.parameter["firstname"] = ""
        kc.save( os.path.join(genpath, "{:03}_{:03}.png".format(epoch,i)) )
