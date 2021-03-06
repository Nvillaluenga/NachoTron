import tensorflow as tf
from hparams import hparams
from feeder import Feeder
from encoder import Encoder

# Bahdanau Attention layer
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)
    self.batch_size = hparams['batch_size']

  def call(self, query, values):
    query_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))
    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = tf.reduce_sum(
      (attention_weights * values), axis=1)

    return context_vector, attention_weights
    
  def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, 2*hparams["enc_lstm_units"]))

if __name__ == "__main__":
  feeder = Feeder()
  sentences, audio_tittles = feeder.create_dataset()
  encoder = Encoder(hparams, True, "Test")
  input_batch, _ = feeder.get_batch(encoder.batch_size, (sentences, audio_tittles))
  sample_hidden = encoder.initialize_hidden_state()
  sample_output, _, _, _, _ = encoder.call(input_batch, sample_hidden)

  attention_layer = BahdanauAttention(10) # 10 Units as input tensor shape???
  sample_hidden = attention_layer.initialize_hidden_state()
  
  print(f"Hidden {sample_hidden.shape}, enc output {sample_output.shape}")
  context_vector, attention_weights = attention_layer(sample_hidden, sample_output)

  print("Attention result shape: (batch size, units) {}".format(context_vector.shape))
  print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))