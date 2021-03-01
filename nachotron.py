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
    
    self.feeder = Feeder()
    self.encoder = Encoder(hparams, is_training, scope="NachoTronEncoder")
    self.decoder = Decoder(hparams)
    self.postnet = Postnet(hparams)
    self.hparams = hparams
    
  def call(self, inputs, previous_step_output, encoder_hidden=None):
    x, _, _, _, _ = self.encoder(inputs, encoder_hidden)
    x, stop_token = self.decoder(x, previous_step_output)
    x = self.postnet(x)
    return x, stop_token
  
  def update_dataset(self, data_path, metadata_file = None, mel_folder = None):
    """Used to change the path of the dataset, so next time the feeder create a dataset"""
    self.feeder.data_path = data_path
    self.feeder.metadata_file = self.hparams['default_metadata_file'] if metadata_file == None else metadata_file
    self.feeder.mel_folder = self.hparams['default_mel_folder'] if mel_folder == None else mel_folder
  
  def get_batch(self, sentences, audio_tittles):
    return self.feeder.get_batch(self.encoder.batch_size, (sentences, audio_tittles))
    
if __name__ == "__main__":
  print("Create NachoTron")
  nachotron = NachoTron(hparams, True)
  print("Get sentenceces from internal feeder")
  sentences, audio_tittles = nachotron.feeder.create_dataset()
  input_batch, _ = nachotron.get_batch(sentences, audio_tittles)
  encoder_initial_hidden_state = nachotron.encoder.initialize_hidden_state()
  previous_step_output = nachotron.decoder.initial_zero_output()
  print("Call NachoTron")
  output, stop_token = nachotron(input_batch, previous_step_output, encoder_initial_hidden_state)
  print(f"Input Shape: {input_batch.shape}")  
  print(f"Output Shape: {output.shape}")