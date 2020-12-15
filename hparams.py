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

  #Decoder parameters
  "dec_prenet_units" : 256, # number of units of each of the 2 prenet layers
  "dec_prenet_dropout_rate" : 0.5, # dropout for the decoder prenet
  "dec_prenet_activation" : 'relu',
  "dec_lstm_units" : 512, # for the 2 stack LSTM layers of the decoder
  "dec_lstm_activation" : 'tanh',
  "dec_lstm_zoneout" : 0.1,
  "dec_lstm_mi" : (1, 0.5, 0.5),
  
  
  #Mel spectrogram
  "num_mels" : 80, #Number of mel-spectrogram channels and local conditioning dimensionality
  "num_freq" : 1025, # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
  "rescale" : True, #Whether to rescale audio prior to preprocessing
  "rescaling_max" : 0.999, #Rescaling value

  #Limits
  "min_level_db" : -100,
  "ref_level_db" : 20,
  "fmin" : 55, #Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
  "fmax" : 7600, #To be increased/reduced depending on data.
  
  #train samples of lengths between 3sec and 14sec are more than enough to make a model capable of generating consistent speech.
  "clip_mels_length" : True, #For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, also consider clipping your samples to smaller chunks)
  "max_mel_frames" : 900,  #Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3 and still getting OOM errors.

  "n_fft" : 2048, #Extra window size is filled with 0 paddings to match this parameter
  "hop_size" : 275, #For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
  "win_size" : 1100, #For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
  "sample_rate" : 22050, #22050 Hz (corresponding to ljspeech dataset) (sox --i <filename>)
  "frame_shift_ms" : None, #Can replace hop_size parameter. (Recommended: 12.5)
  "magnitude_power" : 2., #The power of the spectrogram magnitude (1. for energy, 2. for power)

  #M-AILABS (and other datasets) trim params (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
  "trim_silence" : True, #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
  "trim_fft_size" : 2048, #Trimming window size
  "trim_hop_size" : 512, #Trimmin hop length
  "trim_top_db" : 40, #Trimming db difference from reference db (smaller==harder trim.)
 
   #Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
  "preemphasize" : True, #whether to apply filter
  "preemphasis" : 0.97, #filter coefficient.
  "input_type" :"raw", #Raw has better quality but harder to train. mulaw-quantize is easier to train but has lower quality.
  
	#Mel and Linear spectrograms normalization/scaling and clipping
	"signal_normalization" : True, #Whether to normalize mel spectrograms to some predefined range (following below parameters)
	"allow_clipping_in_normalization" : True, #Only relevant if mel_normalization = True
	"symmetric_mels" : True, #Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
	"max_abs_value" : 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion, 
																										  #not too small for fast convergence)
	"normalize_for_wavenet" : True, #whether to rescale to [0, 1] for wavenet. (better audio quality)
	"clip_for_wavenet" : True, #whether to clip [-max, max] before training/synthesizing with wavenet (better audio quality)
	"wavenet_pad_sides" : 1, #Can be 1 or 2. 1 for pad right only, 2 for both sides padding.

}

def hparams_debug_string():
  hp = [f'  {key}: {value}'  for key, value in sorted(hparams.items()) if key != 'sentences']
  return 'Hyperparameters:\n' + '\n'.join(hp)
  