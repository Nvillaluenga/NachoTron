import numpy as np
import tensorflow as tf

# Default hyperparameters
hparams = {
	#Encoder parameters
	"enc_conv_num_layers" : 3, # number of encoder convolutional layers
	"enc_conv_kernel_size" : (5, ), # size of encoder convolution filters for each layer
	"enc_conv_filters" : 512, # number of encoder convolutions filters for each layer
	"enc_conv_activation" : 'relu',
	"enc_lstm_units" : 256, # number of lstm units for each direction (forward and backward)
	"enc_lstm_activation" : 'tanh', 
	"char_size" : 79, # update this after reading all the datasets.
	"char_embedding_dim" : 256,
	"batch_size" : 64,
	"kernel_size" : 3, # the number of inputs to consider in dilated convolutions.
	"enc_dropout_rate" : 0.5, # dropout rate for all convolutional layers

  "dec_prenet_units" : 256, # number of units of each of the 2 prenet layers
  "dec_prenet_dropout_rate" : 0.5, # dropout for the decoder prenet
  "dec_prenet_activation" : 'relu',
	"dec_lstm_units" : 512, # for the 2 stack LSTM layers of the decoder
 	"dec_lstm_activation" : 'tanh',
  "dec_lstm_zoneout" : 0.1,
  "dec_lstm_mi" : (1, 0.5, 0.5),
}

def hparams_debug_string():
	hp = [f'  {key}: {value}'  for key, value in sorted(hparams.items()) if key != 'sentences']
	return 'Hyperparameters:\n' + '\n'.join(hp)
	