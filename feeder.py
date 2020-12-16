import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from hparams import hparams
from os import path
from io import open

class Feeder():
  default_path = 'D:\\Nacho\\Facultad\\Proyecto Final\\Nachotron\\datasets\\training_data'
  default_metada_file = 'train.txt'
  default_mel_folder = 'mels'

  def __init__(self, data_path = default_path, metada_file = default_metada_file, mel_folder = default_mel_folder):
    super(Feeder, self).__init__()
    self.data_path = data_path
    self.metada_file = metada_file
    self.mel_folder = mel_folder

  def create_dataset(self, num_examples = -1):
    """Create a dataset with all the paths passed"""
    lines = [line for line in open(path.join(self.data_path, self.metada_file), encoding='UTF-8').read().strip().split("\n")]
    sentences = np.array([line.split('|')[-1] for line in lines[:num_examples]])
    mel_identifier = np.array([line.split('|')[1] for line in lines[:num_examples]])
    letters = sorted(set( ''.join(sentences).lower() ))
    self.char2idx = {u:i for i, u in enumerate(letters)}
    self.idx2char = np.array(letters)
    sentences = self.pad_sequences(sentences)
    self.dataset = (sentences, mel_identifier)
    return sentences, mel_identifier

  def pad_sequences(self, sequences, pad=' '):
    max_len_string = len(max(sequences, key=len))
    self.max_len_string = max_len_string
    return np.array([string.ljust(max_len_string, pad) for string in sequences])

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
    return ("".join([self.idx2char[idx] for idx in vector])).strip()

  def get_batch(self, batch_size, dataset = None):
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
    input_batch = np.reshape(input_batch, (batch_size, self.max_len_string))
    output_batch = np.array([audio_identifier[idx] for idx in indexes])
    return input_batch, output_batch

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
  print(f'\nMel spectogram of sentence {feeder.devectorize_sentence(input_batch[0])}\n\t-Shape: {mel.shape}\n{mel}')
  feeder.plot_mel(mel = mel, title = feeder.devectorize_sentence(input_batch[0]))