import tensorflow as tf
from hparams import hparams
from feeder import Feeder
from encoder import Encoder
from decoder import Decoder

class Postnet(tf.keras.Model):

  def __init__(self, hparams):
    """Nachotron Postnet"""
    super(Postnet, self).__init__()
    
    postnet_kernel_size = hparams["dec_postnet_kernel_size"]
    postnet_filters = hparams["dec_postnet_filters"]
    postnet_activation = hparams["dec_postnet_activation"]
    self.postnet_no_layers = hparams["dec_postnet_conv_num_layers"]

    self.frame_projection_units = hparams["num_mels"] * hparams["dec_outputs_per_step"]
    frame_projection_activation = hparams["dec_frame_projection_activation"]
    
    self.postnet = [tf.keras.layers.Conv1D(
      filters = postnet_filters,
      kernel_size = postnet_kernel_size,
      activation = postnet_activation,
      padding = 'same') for _ in range(self.postnet_no_layers-1)]
    self.postnet_normalization = [tf.keras.layers
      .BatchNormalization() for _ in range (self.postnet_no_layers-1)]
    self.postnet_final = tf.keras.layers.Conv1D(
      filters = postnet_filters,
      kernel_size = postnet_kernel_size,
      activation = None,
      padding = 'same')

    self.residual_frame_projection = tf.keras.layers.Dense(
      units = self.frame_projection_units,
      activation = frame_projection_activation)

  def call(self, decoder_output):
    residual_x = tf.expand_dims(decoder_output, 1)
    for i in range(self.postnet_no_layers-1):
      residual_x = self.postnet[i](residual_x)
      residual_x = self.postnet_normalization[i](residual_x)
    residual_x = self.postnet_final(residual_x)
    residual_x = self.residual_frame_projection(residual_x)

    # [batch_size, 1, units] => [batch_size, units]
    residual_x = tf.reshape(residual_x, (-1, residual_x.shape[2]))
    
    return residual_x

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
  
  print("Create Postnet")
  postnet = Postnet(hparams)

  print("Call Decoder and Postnet")
  previous_frame_projection = tf.zeros((decoder.batch_size, decoder.frame_projection_units))
  frame_projection, stop_token = decoder(previous_frame_projection, encoder_output)

  residual_frame_projection = postnet(frame_projection)

  frame_projection += residual_frame_projection

  print(f"Encoder Output Shape: {encoder_output.shape}")
  print(f"Stop Token Shape: {stop_token.shape}")
  print(f"Residual Frame projection Shape: {residual_frame_projection.shape}")
  print(f"Frame projection Shape: {frame_projection.shape}")
