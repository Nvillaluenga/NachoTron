import tensorflow as tf
import numpy as np
from hparams import hparams
from .layers.lstm import LSTM

class Decoder(tf.keras.Model):
  
  def __init__(self, hparams, is_training, scope):
    """Nachotron Encoder"""
    super(Decoder, self).__init__()
    batch_size = hparams['batch_size']
    
    prenet_units = hparams['dec_prenet_units']
    prenet_activation = hparams['dec_prenet_activation']
    prenet_dropout = hparams['dec_prenet_dropout_rate']
    
    lstm_units = hparams['dec_lstm_units']
    lstm_activation = hparams['dec_lstm_activation']
    lstm_zoneout = hparams['dec_lstm_zoneout']
    lstm_mi = hparams['dec_lstm_mi']
    
    # Create the different layers of the encoder
    self.prenet_1 = tf.keras.layers.Dense(
      units = prenet_units,
      activation = prenet_activation)
    self.drop_out_1 = tf.keras.layers.Dropout(prenet_dropout)
    self.prenet_2 = tf.keras.layers.Dense(
      units = prenet_units,
      activation = prenet_activation)
    self.drop_out_2 = tf.keras.layers.Dropout(prenet_dropout)
    
    self.lstm_1 = LSTM(
      units = lstm_units,
      zoneout_h = lstm_zoneout,
      zoneout_c = lstm_zoneout,
      mi = lstm_mi,
      activation = lstm_activation,
      return_sequences = True,
      return_state = True) 
    self.lstm_2 = LSTM(
      units = lstm_units,
      zoneout_h = lstm_zoneout,
      zoneout_c = lstm_zoneout,
      mi = lstm_mi,
      activation = lstm_activation,
      return_sequences = True,
      return_state = True)
    
    self.stop_prediction = tf.keras.layers.Dense(
      units = hparams["dec_stop_token_units"],
      activation = hparams["dec_stop_token_activation"])

    self.frame_projection = tf.keras.layers.Dense(
      units = hparams["num_mels"] * hparams["dec_outputs_per_step"],
      activation = hparams["dec_frame_projection_activation"])
    

    
