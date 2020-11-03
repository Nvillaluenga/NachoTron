import tensorflow as tf
import numpy as np
import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from Nachotron.hparams import hparams
from Nachotron.feeder import feeder
import io

class Encoder(tf.keras.Model):

      def __init__(self, hparams, is_training, scope):
            """Nachotron Encoder"""
            super(Encoder, self).__init__()
            self.scope = 'enc_conv_layers' if scope is None else scope # Names are given when calling them
            self.drop_rate = hparams['enc_dropout_rate'] if is_training else 0
            self.batch_size = hparams['batch_size']
            self.lstm_units = hparams['enc_lstm_units']
            filters = hparams['enc_conv_filters']
            kernel_size = hparams['enc_conv_kernel_size']
            self.activation = hparams['enc_conv_activation']
            input_shape = (self.batch_size, None, None) # (Batch_size, time_steps, input vector length)
            
            # Layers Here
            # Generate an embeddin for each char so it's more meaningfull than a integer
            self.character_embedding = tf.keras.layers.Embedding(
                  hparams['char_size'],
                  hparams['char_embedding_dim'])
            self.conv_1 = tf.keras.layers.Conv1D(
                  filters = filters,
                  kernel_size = kernel_size,
                  activation = self.activation,
                  padding = 'same',
                  input_shape = input_shape)
            self.conv_2 = tf.keras.layers.Conv1D(
                  filters = filters,
                  kernel_size = kernel_size,
                  activation = self.activation,
                  padding = 'same')
            self.drop_out = tf.keras.layers.Dropout(self.drop_rate)
            self.conv_3 = tf.keras.layers.Conv1D(
                  filters = filters,
                  kernel_size = kernel_size,
                  activation = self.activation,
                  padding = 'same')
            self.batch_normalization = tf.keras.layers.BatchNormalization()
            self.bidirectional_LSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                  units = self.lstm_units,
                  activation = hparams['enc_lstm_activation'],
                  return_sequences = True,
                  return_state = True))

      def call(self, inputs, hidden=None):
            x = self.character_embedding(inputs)
            x = self.conv_1(x)
            x = self.conv_2(x)
            x = self.drop_out(x)
            x = self.conv_3(x)
            x = self.batch_normalization(x)
            output, forward_h, forward_c, backward_h, backward_c =  self.bidirectional_LSTM(x, initial_state=hidden)
            return output, forward_h, forward_c, backward_h, backward_c

      def initialize_hidden_state(self):
            return [tf.zeros((self.batch_size, self.lstm_units)) for i in range(4)]

  
if __name__ == "__main__":
      print("Create feeder")
      feeder = feeder()
      sentences, audio_tittles = feeder.create_dataset()
      print("Nachotron Encoder test:")
      encoder = Encoder(hparams, True, "Test")
      input_batch, _ = feeder.get_batch((sentences, audio_tittles), encoder.batch_size)
      encoder.build(input_batch.shape)
      print(encoder.summary())
      print(f"\nInput batch shape: {input_batch.shape}")
      sample_hidden = encoder.initialize_hidden_state()
      # print (f'Encoder Hidden state shape: (batch size, units) {sample_hidden.shape}')
      output, _, _, _, _ = encoder.call(input_batch, sample_hidden)
      print(f'\nOutput shape: {output.shape}')
      print(f'\nOutput batch: {output}')

      
      
      # sample_output, forward_h, forward_c, backward_h, backward_c = encoder(example_input_batch, sample_hidden)
      # print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))
      # print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))