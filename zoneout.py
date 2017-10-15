# from ipywidgets import interact
import tensorflow as tf

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope

# Wrapper for the TF RNN cell
# For an LSTM, the 'cell' is a tuple containing state and cell
# We use TF's dropout to implement zoneout
class ZoneoutWrapper(tf.nn.rnn_cell.RNNCell):
  """Operator adding zoneout to all states (states+cells) of the given cell."""

  def __init__(self, cell, state_zoneout_prob, is_training=True, seed=None):
    if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
      raise TypeError("The parameter cell is not an RNNCell.")
    if (isinstance(state_zoneout_prob, float) and
        not (state_zoneout_prob >= 0.0 and state_zoneout_prob <= 1.0)):
      raise ValueError("Parameter zoneout_prob must be between 0 and 1: %d"
                       % zoneout_prob)
    self._cell = cell
    self._zoneout_prob = state_zoneout_prob
    self._seed = seed
    self.is_training = is_training

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    output, new_state = self._cell(inputs, state, scope)
    if self.is_training:
        new_state = (1 - self._zoneout_prob) * tf.nn.dropout(
                      new_state - state, (1 - self._zoneout_prob), seed=self._seed) + state
    else:
        new_state = self._zoneout_prob * state + (1 - self._zoneout_prob) * new_state
    return output, new_state
