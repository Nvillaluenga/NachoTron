import tensorflow as tf
import numpy as np
from hparams import hparams
from attention import BahdanauAttention
from layers.lstm import LSTM
from feeder import Feeder
from encoder import Encoder

class Decoder(tf.keras.Model):
  
  def __init__(self, hparams):
    """Nachotron Decoder"""
    super(Decoder, self).__init__()
    
    self.batch_size = hparams['batch_size']
    
    prenet_units = hparams['dec_prenet_units']
    prenet_activation = hparams['dec_prenet_activation']
    prenet_dropout = hparams['dec_prenet_dropout_rate']
    
    self.lstm_units = hparams['dec_lstm_units']
    lstm_activation = hparams['dec_lstm_activation']
    lstm_zoneout = hparams['dec_lstm_zoneout']
    lstm_mi = hparams['dec_lstm_mi']

    self.frame_projection_units = hparams["num_mels"] * hparams["dec_outputs_per_step"]
    frame_projection_activation = hparams["dec_frame_projection_activation"]

    # Create the different layers of the encoder
    self.prenet_1 = tf.keras.layers.Dense(
      units = prenet_units,
      activation = prenet_activation)
    self.drop_out_1 = tf.keras.layers.Dropout(prenet_dropout)
    self.prenet_2 = tf.keras.layers.Dense(
      units = prenet_units,
      activation = prenet_activation)
    self.drop_out_2 = tf.keras.layers.Dropout(prenet_dropout)
    
    self.attention = BahdanauAttention(self.lstm_units)

    self.lstm_1 = LSTM(
      units = self.lstm_units,
      zoneout_h = lstm_zoneout,
      zoneout_c = lstm_zoneout,
      mi = lstm_mi,
      activation = lstm_activation,
      return_sequences = True,
      return_state = True) 
    self.lstm_2 = LSTM(
      units = self.lstm_units,
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
      units = self.frame_projection_units,
      activation = frame_projection_activation)

  def initialize_hidden_state(self):
    return [[tf.zeros((self.batch_size, self.lstm_units)) for _ in range(2)] for _ in range(3)]

  def call(self, previous_step_decoder_output, encoder_ouput):
    """
    Args:
    - previous_step_decoder_output: self explanatory
    - encoder_output: self explanatory
    """
    # Prenet call, Used as information botleneck to aid attention convergence
    x = self.prenet_1(previous_step_decoder_output)
    x = self.drop_out_1(x)
    x = self.prenet_2(x)
    x = self.drop_out_2(x)

    context_vector, _ = self.attention(x, encoder_ouput)

    x = tf.concat([tf.expand_dims(context_vector, 1), tf.expand_dims(x, 1)], axis=-1)

    # LSTM call
    x, _, _ = self.lstm_1(x)
    x, _, _ = self.lstm_2(x)

    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # When the prefiction is more than 0.5 we should stop, how do we stop?
    stop_token = self.stop_prediction(x)

    # Predict the mel spectogram as a linear projection
    x = self.frame_projection(x)

    # # [batch_size, 1, units] => [batch_size, units]
    x = tf.reshape(x, (-1, x.shape[2]))
    stop_token = tf.reshape(stop_token, (-1, stop_token.shape[2]))

    return x, stop_token

if __name__ == "__main__":
  print("Create Feeder")
  feeder = Feeder()
  sentences, audio_tittles = feeder.create_dataset()
  print("Create Encoder")
  encoder = Encoder(hparams, True, "Test")
  input_batch, _ = feeder.get_batch(encoder.batch_size, (sentences, audio_tittles))
  encoder.build(input_batch.shape)
  sample_hidden = encoder.initialize_hidden_state()
  encoder_output, _, _, _, _ = encoder(input_batch, sample_hidden)

  print("Create Decoder")
  decoder = Decoder(hparams)

  print("Call Decoder")
  previous_frame_projection = tf.zeros((decoder.batch_size, decoder.frame_projection_units))
  frame_projection, stop_token = decoder(previous_frame_projection, encoder_output)

  frame_projection, stop_token = decoder(frame_projection, encoder_output)

  print(f"Encoder Output Shape: {encoder_output.shape}")
  print(f"Stop Token Shape: {stop_token.shape}")
  print(f"Frame projection Shape: {frame_projection.shape}")