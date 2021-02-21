#/usr/bin/env python

import os
from create_dataset import category_to_onehot
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.losses import mean_squared_error, mean_squared_logarithmic_error
from tensorflow.keras.callbacks import LambdaCallback
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class VariationalAutoEncoder:
    def __init__(self, original_dim, latent_dim):
        self.original_dim = original_dim
        #self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim

    def build_encoder(self):
        x = Input(shape=(self.original_dim, ))
        z_mean = Dense(self.latent_dim, activation='linear')(x)
        z_logvar = Dense(self.latent_dim, activation='linear')(x)
        return Model(x, [z_mean, z_logvar])
    
    def build_decoder(self):
        z_mean = Input(shape=(self.latent_dim ))
        z_logvar = Input(shape=(self.latent_dim, ))
        z = Lambda(self.sampling, output_shape=(self.latent_dim))([z_mean, z_logvar])
        x_decoded_mean = Dense(self.original_dim, activation='sigmoid')(z)

        return Model([z_mean, z_logvar], x_decoded_mean)
    
    @tf.function
    def sampling(self, args):
        z_mean, z_logvar = args
        epsilon = K.random_normal(shape=(self.latent_dim,))
        return z_mean + K.exp(z_logvar*0.5) * epsilon
    
    def build_vae(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        _, encoder_mean, encoder_logvar = encoder.layers

        x = Input(shape=(self.original_dim, ))
        z_mean = encoder_mean(x)
        z_logvar = encoder_logvar(x)

        _, _, decoder_lambda, decoder_dense = decoder.layers
        z = decoder_lambda([z_mean, z_logvar])
        x_decoded_mean = decoder_dense(z)
        return Model(x, x_decoded_mean)
    
    @tf.function
    def vae_loss(self, x, x_decoded_mean):
        _, encoder_mean, encoder_logvar = self.encoder.layers
        z_mean = encoder_mean(x)
        z_logvar = encoder_logvar(x)

        # 1項目の計算
        latent_loss = - 0.5 * K.sum(1 + z_logvar - K.square(z_mean) - K.exp(z_logvar), axis=-1)
        # 2項目の計算
        reconst_loss = K.mean(mean_squared_error(x, x_decoded_mean), axis=-1)
        return latent_loss + reconst_loss

    @tf.function
    def sample_variance(self, x, y):
        _, _, decoder_lambda, decoder_dense = self.decoder.layers
        input1 = K.random_normal(shape=(1000, self.latent_dim))
        input2 = K.random_normal(shape=(1000, self.latent_dim))

        z = decoder_lambda([input1, input2])
        x_decoded_mean = decoder_dense(z)

        return K.mean(K.std(x_decoded_mean, axis=0))
    
    @tf.function
    def data_variance(self, x, y):
        _, encoder_mean, encoder_logvar = self.encoder.layers
        z_mean = encoder_mean(x)
        z_logvar = encoder_logvar(x)

        return K.mean(K.std(z_mean, axis=0))

    def model_compile(self, model):
        model.compile(optimizer='adamax', loss=self.vae_loss, metrics=[self.sample_variance, self.data_variance], run_eagerly=True)

def main():
    df = pd.read_csv("./kk_charas.csv", index_col=0)
    df = category_to_onehot(df)
    df = df.clip(0, 1)

    train = df.values

    vae = VariationalAutoEncoder(df.shape[1], 800)
    encoder = vae.build_encoder()
    decoder = vae.build_decoder()
    model = vae.build_vae(encoder, decoder)
    vae.model_compile(model)

    t = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    dirpath = os.path.join("./vae_models", t)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    def save_model(epoch, logs):
        if epoch % 5 != 0:
            return
        
        filename = "{:03}_%s.%s".format(epoch)
        filepath = os.path.join(dirpath, filename)
        with open(filepath % ("encoder", "json"), "w+") as f:
            f.write(encoder.to_json(indent=2))
        with open(filepath % ("decoder", "json"), "w+") as f:
            f.write(decoder.to_json(indent=2))
        encoder.save_weights(filepath % ("encoder", "h5"))
        decoder.save_weights(filepath % ("decoder", "h5"))

    save_model_callback = LambdaCallback(on_epoch_end=save_model)

    model.fit(
        train,
        train,
        epochs = 100,
        batch_size = 200,
        callbacks = [save_model_callback]
    )

    save_model(100, None)

if __name__=='__main__':
    main()