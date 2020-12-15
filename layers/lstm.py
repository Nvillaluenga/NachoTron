from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import sys

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
  

if __name__ == "__main__":
  lstm = LSTM(
    units = 10,
    zoneout_h = 0.1,
    zoneout_c = 0.1,
    mi = (1, 0.5, 0.5),
    return_sequences = True,
    return_state = True,
    name = 'example',
    input_shape=(10, 8))
  inputs = tf.random.normal([32, 10, 8])
  print(f"Input shape {inputs.shape}") # [batch_size, time_steps, feature]
  output, final_memory_state, final_carry_state = lstm(inputs)
  print(f"Output shape {tf.shape(output)} \nOutput: {output}") # [batch_size, time_steps, units]
  print(f"final_memory_state {tf.shape(final_memory_state)}")
  print(f"final_carry_state {tf.shape(final_carry_state)}")
  
  ##### TEST IT WORKS FOR BUILDING
  if (False):
    TRAINING_SIZE = 50000
    DIGITS = 3
    REVERSE = True

    # Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
    # int is DIGITS.
    MAXLEN = DIGITS + 1 + DIGITS
    class CharacterTable:
        """Given a set of characters:
        + Encode them to a one-hot integer representation
        + Decode the one-hot or integer representation to their character output
        + Decode a vector of probabilities to their character output
        """

        def __init__(self, chars):
            """Initialize character table.
            # Arguments
                chars: Characters that can appear in the input.
            """
            self.chars = sorted(set(chars))
            self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
            self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

        def encode(self, C, num_rows):
            """One-hot encode given string C.
            # Arguments
                C: string, to be encoded.
                num_rows: Number of rows in the returned one-hot encoding. This is
                    used to keep the # of rows for each data the same.
            """
            x = np.zeros((num_rows, len(self.chars)))
            for i, c in enumerate(C):
                x[i, self.char_indices[c]] = 1
            return x

        def decode(self, x, calc_argmax=True):
            """Decode the given vector or 2D array to their character output.
            # Arguments
                x: A vector or a 2D array of probabilities or one-hot representations;
                    or a vector of character indices (used with `calc_argmax=False`).
                calc_argmax: Whether to find the character index with maximum
                    probability, defaults to `True`.
            """
            if calc_argmax:
                x = x.argmax(axis=-1)
            return "".join(self.indices_char[x] for x in x)


    # All the numbers, plus sign and space for padding.
    chars = "0123456789+ "
    ctable = CharacterTable(chars)

    questions = []
    expected = []
    seen = set()
    print("Generating data...")
    while len(questions) < TRAINING_SIZE:
        f = lambda: int(
            "".join(
                np.random.choice(list("0123456789"))
                for i in range(np.random.randint(1, DIGITS + 1))
            )
        )
        a, b = f(), f()
        # Skip any addition questions we've already seen
        # Also skip any such that x+Y == Y+x (hence the sorting).
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        # Pad the data with spaces such that it is always MAXLEN.
        q = "{}+{}".format(a, b)
        query = q + " " * (MAXLEN - len(q))
        ans = str(a + b)
        # Answers can be of maximum size DIGITS + 1.
        ans += " " * (DIGITS + 1 - len(ans))
        if REVERSE:
            # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
            # space used for padding.)
            query = query[::-1]
        questions.append(query)
        expected.append(ans)
    print("Total questions:", len(questions))

    print("Vectorization...")
    x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, MAXLEN)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, DIGITS + 1)

    # Shuffle (x, y) in unison as the later parts of x will almost all be larger
    # digits.
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # Explicitly set apart 10% for validation data that we never train over.
    split_at = len(x) - len(x) // 10
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]

    print("Training Data:")
    print(x_train.shape)
    print(y_train.shape)

    print("Validation Data:")
    print(x_val.shape)
    print(y_val.shape)

    print("Build model...")
    num_layers = 1  # Try to add more LSTM layers!

    model = keras.Sequential()
    # "Encode" the input sequence using a LSTM, producing an output of size 128.
    # Note: In a situation where your input sequences have a variable length,
    # use input_shape=(None, num_feature).
    model.add(LSTM(
      units = 128,
      zoneout_h = 0.1,
      zoneout_c = 0.1,
      mi = (1, 0.5, 0.5),
      name = 'example',
      input_shape=(MAXLEN, len(chars))))
    # As the decoder RNN's input, repeatedly provide with the last output of
    # RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
    # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
    model.add(layers.RepeatVector(DIGITS + 1))
    # The decoder RNN could be multiple layers stacked or a single layer.
    for _ in range(num_layers):
        # By setting return_sequences to True, return not only the last output but
        # all the outputs so far in the form of (num_samples, timesteps,
        # output_dim). This is necessary as TimeDistributed in the below expects
        # the first dimension to be the timesteps.
        model.add(layers.LSTM(128, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step
    # of the output sequence, decide which character should be chosen.
    model.add(layers.Dense(len(chars), activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    epochs = 30
    batch_size = 32


    # Train the model each generation and show predictions against the validation
    # dataset.
    for epoch in range(1, epochs):
        print()
        print("Iteration", epoch)
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=1,
            validation_data=(x_val, y_val),
        )
        # Select 10 samples from the validation set at random so we can visualize
        # errors.
        for i in range(10):
            ind = np.random.randint(0, len(x_val))
            rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
            preds = np.argmax(model.predict(rowx), axis=-1)
            q = ctable.decode(rowx[0])
            correct = ctable.decode(rowy[0])
            guess = ctable.decode(preds[0], calc_argmax=False)
            print("Q", q[::-1] if REVERSE else q, end=" ")
            print("T", correct, end=" ")
            if correct == guess:
                print("☑ " + guess)
            else:
                print("☒ " + guess)