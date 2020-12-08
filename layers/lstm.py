import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

###################### Helper Functions
def layer_normalization(x, gain, bias, epsilon=1e-5):
    mean, std = tf.nn.moments(x, [1], keep_dims=True)
    x_normed = (x - mean) / K.sqrt(std + epsilon) * gain + bias
    return x_normed

def k_init(k):
  def init(shape, dtype='float32', name=None):
    return K.variable(k*np.ones(shape), dtype=dtype, name=name)
  return init
  
def zoneout(level, h_tm1, h, noise_shape):
  '''Apply a zoneout function to preserve a fraction of values from h_tm1 in h.'''
  h_diff = h - h_tm1
  h = K.in_train_phase(K.dropout(h_diff,
                                  level,
                                  noise_shape=noise_shape), h_diff)
  h = h * (1. - level) + h_tm1
  return h
###################### /Helper Functions

class LSTM(tf.keras.layers.LSTM):
  """
  # Arguments
      ln: None, list of float or list of list of floats. Determines whether will apply LN or not. If list of floats, the same init will be applied to every LN; otherwise will be individual
      mi: list of floats or None. If list of floats, the multiplicative integration will be active and initialized with these values.
      zoneout_h: float between 0 and 1. Fraction of the hidden/output units to maintain their previous values.
      zoneout_c: float between 0 and 1. Fraction of the cell units to maintain their previous values.
  # References
      - [Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations](https://arxiv.org/abs/1606.01305)
      - [Multiplicative integration: On Multiplicative Integration with Recurrent Neural Networks](https://arxiv.org/pdf/1606.06630.pdf)
  """
  def __init__(self, units, zoneout_h=0., zoneout_c=0., layer_norm=None, mi=None, **kwargs):
    self.output_dim = units
    super(LSTM, self).__init__(units, **kwargs)

    self.layer_norm = layer_norm
    self.mi = mi

    self.zoneout_c = zoneout_c
    self.zoneout_h = zoneout_h

    if self.zoneout_h or self.zoneout_c:
      self.uses_learning_phase = True

    self.consume_less = 'gpu'

  def build(self, input_shape):
    super(LSTM, self).build(input_shape)
    if self.mi is not None:
      alpha_init, beta1_init, beta2_init = self.mi
      print('#############################################')
      print(f'{alpha_init}, {beta1_init}, {beta2_init}')

      self.mi_alpha = self.add_weight(
        shape = (4 * self.output_dim, ),
        initializer=k_init(alpha_init),
        name='{}_mi_alpha'.format(self.name))
      self.mi_beta1 = self.add_weight(
        shape = (4 * self.output_dim, ),
        initializer=k_init(beta1_init),
        name='{}_mi_beta1'.format(self.name))
      self.mi_beta2 = self.add_weight(
        shape = (4 * self.output_dim, ),
        initializer=k_init(beta2_init),
        name='{}_mi_beta2'.format(self.name))

    if self.layer_norm is not None:
      ln_gain_init, ln_bias_init = self.layer_norm

      self.layer_norm_params = {}
      for n, i in {'Uh': 4, 'Wx': 4, 'new_c': 1}.items():

        gain = self.add_weight(
          (i*self.output_dim, ),
          initializer=k_init(ln_gain_init),
          name='%s_ln_gain_%s' % (self.name, n))
        bias = self.add_weight(
          (i*self.output_dim, ),
          initializer=k_init(ln_bias_init),
          name='%s_ln_bias_%s' % (self.name, n))

        self.layer_norm_params[n] = [gain, bias]

  def _layer_norm(self, x, param_name):
    if self.layer_norm is None:
      return x

    gain, bias = self.layer_norm_params[param_name]

    return layer_normalization(x, gain, bias)

  def step(self, x, states):
    h_tm1 = states[0]
    c_tm1 = states[1]
    B_U = states[2]
    B_W = states[3]

    Uh = self._layer_norm(K.dot(h_tm1 * B_U[0], self.U), 'Uh')
    Wx = self._layer_norm(K.dot(x * B_W[0], self.W), 'Wx')

    if self.mi is not None:
      z = self.mi_alpha * Wx * Uh + self.mi_beta1 * Uh + self.mi_beta2 * Wx + self.b
    else:
      z = Wx + Uh + self.b

    z_i = z[:, :self.output_dim]
    z_f = z[:, self.output_dim: 2 * self.output_dim]
    z_c = z[:, 2 * self.output_dim: 3 * self.output_dim]
    z_o = z[:, 3 * self.output_dim:]

    i = self.inner_activation(z_i)
    f = self.inner_activation(z_f)
    c = f * c_tm1 + i * self.activation(z_c)
    o = self.inner_activation(z_o)

    if 0 < self.zoneout_c < 1:
      c = zoneout(self.zoneout_c, c_tm1, c, noise_shape=(self.output_dim,))

    # this is returning a lot of nan
    new_c = self._layer_norm(c, 'new_c')

    h = o * self.activation(new_c)
    if 0 < self.zoneout_h < 1:
      h = zoneout(self.zoneout_h, h_tm1, h, noise_shape=(self.output_dim,))

    return h, [h, c]

  def get_config(self):
    config = {
      'layer_norm': self.layer_norm,
      'mi': self.mi,
      'zoneout_h': self.zoneout_h,
      'zoneout_c': self.zoneout_c
    }

    base_config = super(LSTM, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
  
##### TEST IT WORKS FOR BUILDING
if __name__ == "__main__":
  lstm = LSTM(
    units = 10,
    zoneout_h = 0.1,
    zoneout_c = 0.1,
    mi = (1, 0.5, 0.5),
    return_sequences = True,
    return_state = True,
    name = 'example')
  inputs = tf.random.normal([32, 10, 8])
  print(f"Input shape {inputs.shape}") # [batch_size, time_steps, feature]
  output, final_memory_state, final_carry_state = lstm(inputs)
  print(f"Output shape {tf.shape(output)} \nOutput: {output}") # [batch_size, time_steps, units]
  print(f"final_memory_state {tf.shape(final_memory_state)}")
  print(f"final_carry_state {tf.shape(final_carry_state)}")