import tensorflow as tf
import numpy as np
from hparams import hparams

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
    
    # Create the different layers of the encoder
    
    self.prenet_1 = tf.keras.layers.Dense(
      units = prenet_units,
      activation = prenet_activation)
    self.drop_out_1 = tf.keras.layers.Dropout(prenet_dropout)
    self.prenet_2 = tf.keras.layers.Dense(
      units = prenet_units,
      activation = prenet_activation)
    self.drop_out_2 = tf.keras.layers.Dropout(prenet_dropout)
    
    self.lstm_1 = tf.keras.layers.LSTM(
      units = lstm_units,
      activation = lstm_activation,
      return_sequences = True,
      return_state = True)
    
    

    