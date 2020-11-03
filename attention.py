import tensorflow as tf
import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from Nachotron.hparams import hparams
from Nachotron.feeder import feeder
from Nachotron.nachotron import Encoder

# Bahdanau Attention layer
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)
    self.units = units
    self.batch_size = hparams['batch_size']

  def call(self, query, values):
    query_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))
    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
    
  def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, 2*hparams["enc_lstm_units"]))

if __name__ == "__main__":
  feeder = feeder()
  sentences, audio_tittles = feeder.create_dataset()
  encoder = Encoder(hparams, True, "Test")
  input_batch, _ = feeder.get_batch((sentences, audio_tittles), encoder.batch_size)
  encoder.build(input_batch.shape)
  print(encoder.summary())
  sample_hidden = encoder.initialize_hidden_state()
  sample_output, _, _, _, _ = encoder.call(input_batch, sample_hidden)

  attention_layer = BahdanauAttention(10) # 10 Units as input tensor shape???
  sample_hidden = attention_layer.initialize_hidden_state()
  
  print(f"Hidden {sample_hidden.shape}, enc output {sample_output.shape}")
  attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

  print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
  print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))