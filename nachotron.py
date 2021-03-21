from postnet import Postnet
from decoder import Decoder
from encoder import Encoder
from feeder import Feeder
from hparams import hparams
import tensorflow as tf

class NachoTron(tf.keras.Model):
  def __init__(self, hparams, is_training):
    """Nachotron wrapper module"""
    super(NachoTron, self).__init__()
    # Building blocks
    self.feeder = Feeder()
    self.encoder = Encoder(hparams, is_training, scope="NachoTronEncoder")
    self.decoder = Decoder(hparams)
    self.postnet = Postnet(hparams)
    self.hparams = hparams
    # Optimizer
    self.optimizer = tf.keras.optimizers.Adam()
    # Loss
    self.loss_object = tf.keras.losses.MeanSquaredError(
      reduction = tf.keras.losses.Reduction.AUTO,
      name='mean_squared_error')
    self.stop_token_loss_object = tf.keras.losses.MeanSquaredError(
      reduction = tf.keras.losses.Reduction.AUTO,
      name='mean_squared_error')
    
  def call(self, inputs, previous_step_output, encoder_hidden=None):
    x, _ = self.encoder(inputs, encoder_hidden)
    x, stop_token = self.decoder(x, previous_step_output)
    x = self.postnet(x)
    return x, stop_token
  
  def update_dataset(self, data_path, metadata_file = None, mel_folder = None):
    """Used to change the path of the dataset, so next time the feeder create a dataset"""
    self.feeder.data_path = data_path
    self.feeder.metadata_file = self.hparams['default_metadata_file'] if metadata_file == None else metadata_file
    self.feeder.mel_folder = self.hparams['default_mel_folder'] if mel_folder == None else mel_folder
  
  def get_batch(self, sentences, audio_tittles, mel_as_array = True):
    return self.feeder.get_batch(self.encoder.batch_size, (sentences, audio_tittles), mel_as_array=mel_as_array)

  def loss_function(self, real, pred_frame_projection):
    real_stop_token, real_frame_projection = real
    mask = tf.math.logical_not(tf.math.equal(real_stop_token, 0))
    loss_ = self.loss_object(real_frame_projection, pred_frame_projection)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)
  
  def stop_token_loss_function(self, real, pred):
    loss_ = self.stop_token_loss_object(real, pred)
    return tf.reduce_mean(loss_)
  
  @tf.function
  def train_step (self, inp, targ, enc_hidden):
    loss = 0
    with_postnet_loss = 0
    
    with tf.GradientTape() as tape:
      enc_output, enc_hidden = self.encoder(inp, enc_hidden)
      dec_input = self.decoder.initial_zero_output()
      
      for t in range(1, targ.shape[1]):
        frame_projection_targ, stop_token_targ = targ[:, t]
        predictions, stop_token = self.decoder(enc_output, dec_input)
        loss += self.loss_function((frame_projection_targ, stop_token_targ), predictions)
        
        predictions += self.postnet(predictions)
        with_postnet_loss += self.loss_function((frame_projection_targ, stop_token_targ), predictions)
        
        stop_token_loss = self.stop_token_loss_function(stop_token_targ, stop_token)
        
        # Using teacher forcing - Instead of passing the predictions, we pass correct results
        dec_input = tf.expand_dims(frame_projection_targ, 1)

      batch_loss = (loss / int(targ.shape[1]))

      variables = self.encoder.trainable_variables + self.decoder.trainable_variables
      postnet_variables = self.postnet.trainable_variables
      stop_token_variables = self.decoder.stop_prediction.trainable_variables
      
      gradients = tape.gradient(loss, variables)
      self.optimizer.apply_gradients(zip(gradients, variables))
      
      postnet_gradients = tape.gradient(with_postnet_loss, postnet_variables)
      self.optimizer.apply_gradients(zip(postnet_gradients, postnet_variables))

      stop_token_gradients = tape.gradient(stop_token_loss, stop_token_variables)
      self.optimizer.apply_gradients(zip(stop_token_gradients, stop_token_variables))

    return batch_loss
    
if __name__ == "__main__":
  print("Create NachoTron")
  nachotron = NachoTron(hparams, True)
  print("Get sentenceces from internal feeder")
  sentences, audio_tittles = nachotron.feeder.create_dataset()
  input_batch, expected_batch = nachotron.get_batch(sentences, audio_tittles)
  mel_targets, stop_tokens = expected_batch
  encoder_initial_hidden_state = nachotron.encoder.initialize_hidden_state()
  previous_step_output = nachotron.decoder.initial_zero_output()
  print("Call NachoTron")
  output, stop_token = nachotron(input_batch, previous_step_output, encoder_initial_hidden_state)
  print(f"Input Shape: {input_batch.shape}")  
  print(f"Output Shape: {output.shape}")
  print(f"Mel targets: {mel_targets.shape}")
  print(f"Stop tokens: {stop_tokens.shape}")
