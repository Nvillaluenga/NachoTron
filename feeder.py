import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from hparams import hparams
from os import path
from io import open

padding_char = '#'

class Feeder():
  default_path = hparams['default_path']
  default_metadata_file = hparams['default_metadata_file']
  default_mel_folder = hparams['default_mel_folder']

  def __init__(self, data_path = default_path, metadata_file = default_metadata_file, mel_folder = default_mel_folder):
    super(Feeder, self).__init__()
    self.data_path = data_path
    self.metadata_file = metadata_file
    self.mel_folder = mel_folder

  # Need to see a way to mantain the index of the char used, because they may be different between datastes
  def create_dataset(self, num_examples = -1):
    """Create a dataset with all the paths passed"""
    lines = [line for line in open(path.join(self.data_path, self.metadata_file), encoding='UTF-8').read().strip().split("\n")]
    sentences = np.array([line.split('|')[-1] for line in lines[:num_examples]])
    mel_identifier = np.array([line.split('|')[1] for line in lines[:num_examples]])
    letters = set( ''.join(sentences).lower() )
    letters = sorted(letters)
    letters.insert(0, padding_char) # we need a character that we don't use as a the 0 tha's gonna be used as a mask
    self.char2idx = {u:i for i, u in enumerate(letters)}
    self.idx2char = np.array(letters)
    self.dataset = (sentences, mel_identifier)
    return sentences, mel_identifier

  # Deprecated, Changed for tf padding after vectorizing
  # def pad_sequences(self, sequences, pad=padding_char):
  #   max_len_string = len(max(sequences, key=len))
  #   self.max_len_string = max_len_string
  #   return np.array([string.ljust(max_len_string, pad) for string in sequences])

  def vectorize_sentence(self, sentence):
    """
    Change chars in a sentence for their index equivalent of the dataset

    Args:
      - sentence: Sentence to "vectorize"
    Returns:
      - An array with the idx of each char
    """
    return np.array([self.char2idx[char] for char in sentence.lower()])

  def devectorize_sentence(self, vector):
    """
    Change indexes equivalents in an input to the sentence

    Args:
      - vector: vector to convert to sentence
    Returns:
      - The sentence as a string
    """
    return ("".join([self.idx2char[idx] for idx in vector])).strip(padding_char)

  def get_batch(self, batch_size, dataset = None, mel_as_array = False):
    """
    Get batch of size batch_size

    Args:
      - batch_size: size of the batch
      - dataset: Optional, dataset (sentences, mel_files) to get a batch from

    Returns:
      - list input_batch of shape (batch_size, max len string)
      - list output_batch of shape (batch_size)
    """
    if (not self.dataset and not dataset):
      raise "Create a dataset (see: create_dataset) or pass a dataset to use this function"
    dataset = dataset if dataset else self.dataset
    sentences, audio_identifier = dataset
    number_of_examples = sentences.shape[0]
    indexes = np.random.choice(number_of_examples, batch_size)
    input_batch = np.array([self.vectorize_sentence(sentences[idx]) for idx in indexes])
    input_batch = tf.keras.preprocessing.sequence.pad_sequences(input_batch, padding='post')
    output_batch = np.array([audio_identifier[idx] for idx in indexes])
    if (mel_as_array):
      output_batch = np.array([self.get_mel(output) for output in output_batch])
      # output_batch = np.reshape(output_batch, (batch_size, -1))
      stop_tokens = np.array([np.ones(output.shape[0]) for output in output_batch])
      stop_tokens = tf.keras.preprocessing.sequence.pad_sequences(stop_tokens, padding='post')
      output_batch = tf.keras.preprocessing.sequence.pad_sequences(output_batch, padding='post')
      output_batch = (output_batch, stop_tokens)
      
    return input_batch, output_batch
  
  def load_dataset(self, batch_size, limit=-1, dataset = None, mel_as_array = False):
    """
    return a tf dataset of size limit

    Args:
      - batch_size: size of the batch
      - limit: Optional, limit the size of the dataset
      - dataset: Optional, dataset (sentences, mel_files) to get a batch from

    Returns:
      - tf dataset
    """
    if (not self.dataset and not dataset):
      raise "Create a dataset (see: create_dataset) or pass a dataset to use this function"
    dataset = dataset if dataset else self.dataset
    sentences, audio_identifier = dataset
    ouputs = np.array(audio_identifier)
    if (limit != -1):
      max_length = len(sentences)
      sentences = sentences[:min(limit, max_length)]
      ouputs = ouputs[:min(limit, max_length)]
    inputs = np.array([self.vectorize_sentence(sentence) for sentence in sentences])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, padding='post')
    if (mel_as_array):
      ouputs = np.array([self.get_mel(output) for output in ouputs])
      stop_tokens = np.array([np.ones(output.shape[0]) for output in ouputs])
      stop_tokens = tf.keras.preprocessing.sequence.pad_sequences(stop_tokens, padding='post')
      ouputs = tf.keras.preprocessing.sequence.pad_sequences(ouputs, padding='post')
      ouputs = (ouputs, stop_tokens)
      
    dataset = tf.data.Dataset.from_tensor_slices((inputs, ouputs)).shuffle(len(inputs))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset

  def get_mel(self, mel_path):
    return np.load(path.join(self.data_path, self.mel_folder, mel_path))

  def plot_mel(self, mel, title = 'Default', save = False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(title)
    im = ax.imshow(np.rot90(mel), interpolation='none')
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax)

    if (save): plt.savefig(title, format='png')
    else: plt.show()
    plt.close()

if __name__ == "__main__":
  print("Nachotron feeder test:")
  feeder = Feeder()
  sentences, audio_identifier = feeder.create_dataset()
  print(f'\nSentence 1 "{sentences[0]}" tittle 1 "{audio_identifier[0]}"') 
  vectorized_string = feeder.vectorize_sentence("My name is Nacho")
  print(f'\nVectorized string: {vectorized_string}')
  print(f'\nchar2idx:\n\t-Size: {len(feeder.char2idx)}\n{feeder.char2idx}')
  input_batch, output_batch = feeder.get_batch(hparams['batch_size'], (sentences, audio_identifier))
  input_batch, output_batch = feeder.get_batch(hparams['batch_size'])
  print(f'\nInput Batch:\n\t-Shape: {input_batch.shape}\n{input_batch}')
  print(f'\nOutput Batch:\n\t-Shape: {output_batch.shape}\n{output_batch}')
  mel = feeder.get_mel(output_batch[0])
  print(f'\nMel spectogram of sentence: {feeder.devectorize_sentence(input_batch[0])}\n\t-Shape: {mel.shape}\n{mel}')
  feeder.plot_mel(mel = mel, title = feeder.devectorize_sentence(input_batch[0]))
  
  print("Test load dataset:")
  dataset = feeder.load_dataset(hparams['batch_size'], dataset = (sentences, audio_identifier), mel_as_array=True)
  input_batch, output_batch = next(iter(dataset))
  print(f'\nInput Batch:\n\t-Shape: {input_batch.shape}\n{input_batch}')
  ouputs, stop_tokens = output_batch
  print(f'\nOutput Batch:\n\t-Shape: MEL: {ouputs.shape}\nStop Tokens:{stop_tokens.shape}')
  mel = ouputs[0]
  print(f'\nMel spectogram of sentence: {feeder.devectorize_sentence(input_batch[0])}\n\t-Shape: {mel.shape}')
  feeder.plot_mel(mel = mel, title = feeder.devectorize_sentence(input_batch[0]))
  