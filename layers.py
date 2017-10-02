# -*- coding: utf-8 -*-
#/usr/bin/python2

from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import RNNCell
from params import Params
import tensorflow as tf
import numpy as np

'''
attention weights from https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
W_u^Q.shape:    (2 * attn_size, attn_size)
W_u^P.shape:    (2 * attn_size, attn_size)
W_v^P.shape:    (attn_size, attn_size)
W_g.shape:      (4 * attn_size, 4 * attn_size)
W_h^P.shape:    (2 * attn_size, attn_size)
W_v^Phat.shape: (2 * attn_size, attn_size)
W_h^a.shape:    (2 * attn_size, attn_size)
W_v^Q.shape:    (attn_size, attn_size)
'''

def get_attn_params(attn_size,initializer = tf.truncated_normal_initializer):
    '''
    Args:
        attn_size: the size of attention specified in https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
        initializer: the author of the original paper used gaussian initialization however I found xavier converge faster

    Returns:
        params: A collection of parameters used throughout the layers
    '''
    with tf.variable_scope("attention_weights"):
        params = {"W_u_Q":tf.get_variable("W_u_Q",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
                "W_ru_Q":tf.get_variable("W_ru_Q",dtype = tf.float32, shape = (2 * attn_size, 2 * attn_size), initializer = initializer()),
                "W_u_P":tf.get_variable("W_u_P",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
                "W_v_P":tf.get_variable("W_v_P",dtype = tf.float32, shape = (attn_size, attn_size), initializer = initializer()),
                "W_v_P_2":tf.get_variable("W_v_P_2",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
                "W_g":tf.get_variable("W_g",dtype = tf.float32, shape = (4 * attn_size, 4 * attn_size), initializer = initializer()),
                "W_h_P":tf.get_variable("W_h_P",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
                "W_v_Phat":tf.get_variable("W_v_Phat",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
                "W_h_a":tf.get_variable("W_h_a",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
                "W_v_Q":tf.get_variable("W_v_Q",dtype = tf.float32, shape = (attn_size,  2 * attn_size), initializer = initializer())}
        return params

def encoding(word, char, word_embeddings, char_embeddings, scope = "embedding"):
    with tf.variable_scope(scope):
        word_encoding = tf.nn.embedding_lookup(word_embeddings, word)
        char_encoding = tf.nn.embedding_lookup(char_embeddings, char)
        return word_encoding, char_encoding

def apply_dropout(inputs, dropout = Params.dropout, is_training = True):
    '''
    Currently not implemented due to high bias error in the training results
    '''
    if not is_training or Params.dropout is None:
        return inputs
    if isinstance(inputs, RNNCell):
        return tf.contrib.rnn.DropoutWrapper(inputs, output_keep_prob=1.0 - dropout, dtype = tf.float32)
    else:
        return tf.nn.dropout(inputs, keep_prob = 1.0 - dropout)

def bidirectional_GRU(inputs, inputs_len, cell = None, units = Params.attn_size, layers = 1, scope = "Bidirectional_GRU", output = 0, is_training = True, reuse = None):
    '''
    Bidirectional recurrent neural network with GRU cells.

    Args:
        inputs:     rnn input of shape (batch_size, timestep, dim)
        inputs_len: rnn input_len of shape (batch_size, )
        cell:       rnn cell of type RNN_Cell.
        output:     if 0, output returns rnn output for every timestep,
                    if 1, output returns concatenated state of backward and
                    forward rnn.
    '''
    with tf.variable_scope(scope, reuse = reuse):
        if cell is not None:
            (cell_fw, cell_bw) = cell
        else:
            if layers > 1:
                cell_fw = MultiRNNCell([tf.contrib.rnn.GRUCell(units) for _ in range(layers)])
                cell_bw = MultiRNNCell([tf.contrib.rnn.GRUCell(units) for _ in range(layers)])
            else:
                cell_fw = tf.contrib.rnn.GRUCell(units)
                cell_bw = tf.contrib.rnn.GRUCell(units)

        shapes = inputs.get_shape().as_list()
        if len(shapes) > 3:
            inputs = tf.reshape(inputs,(shapes[0]*shapes[1],shapes[2],-1))
            inputs_len = tf.reshape(inputs_len,(shapes[0]*shapes[1],))
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                        sequence_length = inputs_len,
                                                        dtype=tf.float32)
        if output == 0:
            return tf.concat(outputs, 2)
        elif output == 1:
            return tf.reshape(tf.concat(states,1),(Params.batch_size, shapes[1], 2*units))

def pointer_net(passage, passage_len, question, question_len, cell, params, scope = "pointer_network"):
    with tf.variable_scope(scope):
        weights_q, weights_p = params
        shapes = passage.get_shape().as_list()
        initial_state = question_pooling(question, units = Params.attn_size, weights = weights_q, memory_len = question_len, scope = "question_pooling")
        inputs = [passage, initial_state]
        p1_logits = attention(inputs, Params.attn_size, weights_p, memory_len = passage_len, scope = "attention")
        scores = tf.expand_dims(p1_logits, -1)
        attention_pool = tf.reduce_sum(scores * passage,1)
        _, state = cell(attention_pool, initial_state)
        inputs = [passage, state]
        p2_logits = attention(inputs, Params.attn_size, weights_p, memory_len = passage_len, scope = "attention", reuse = True)
        return tf.stack((p1_logits,p2_logits),1)

def attention_rnn(inputs, inputs_len, units, attn_cell, bidirection = True, scope = "gated_attention_rnn", is_training = True):
    with tf.variable_scope(scope):
        if bidirection:
            outputs = bidirectional_GRU(inputs,
                                        inputs_len,
                                        cell = attn_cell,
                                        scope = scope + "_bidirectional",
                                        output = 0,
                                        is_training = is_training)
        else:
            outputs, _ = tf.nn.dynamic_rnn(attn_cell, inputs,
                                            sequence_length = inputs_len,
                                            dtype=tf.float32)
        return outputs

def question_pooling(memory, units, weights, memory_len = None, scope = "question_pooling"):
    with tf.variable_scope(scope):
        shapes = memory.get_shape().as_list()
        V_r = tf.get_variable("question_param", shape = (Params.max_q_len, units), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
        inputs_ = [memory, V_r]
        attn = attention(inputs_, units, weights, memory_len = memory_len, scope = "question_attention_pooling")
        attn = tf.expand_dims(attn, -1)
        return tf.reduce_sum(attn * memory, 1)

def gated_attention(memory, inputs, states, units, params, self_matching = False, memory_len = None, scope="gated_attention"):
    with tf.variable_scope(scope):
        weights, W_g = params
        inputs_ = [memory, inputs]
        states = tf.reshape(states,(Params.batch_size,Params.attn_size))
        if not self_matching:
            inputs_.append(states)
        scores = attention(inputs_, units, weights, memory_len = memory_len)
        scores = tf.expand_dims(scores,-1)
        attention_pool = tf.reduce_sum(scores * memory, 1)
        inputs = tf.concat((inputs,attention_pool),axis = 1)
        g_t = tf.sigmoid(tf.matmul(inputs,W_g))
        return g_t * inputs

def mask_attn_score(score, memory_sequence_length, score_mask_value = -1e8):
    score_mask = tf.sequence_mask(
        memory_sequence_length, maxlen=score.shape[1])
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(score_mask, score, score_mask_values)

def attention(inputs, units, weights, scope = "attention", memory_len = None, reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        outputs_ = []
        for i, (inp,w) in enumerate(zip(inputs,weights)):
            shapes = inp.shape.as_list()
            inp = tf.reshape(inp, (-1, shapes[-1]))
            if w is None:
                w = tf.get_variable("w_%d"%i, dtype = tf.float32, shape = [shapes[-1],Params.attn_size], initializer = tf.contrib.layers.xavier_initializer())
            outputs = tf.matmul(inp, w)
            # Hardcoded attention output reshaping. Equation (4), (8), (9) and (11) in the original paper.
            if len(shapes) > 2:
                outputs = tf.reshape(outputs, (shapes[0], shapes[1], -1))
            elif len(shapes) == 2 and shapes[0] is Params.batch_size:
                outputs = tf.reshape(outputs, (shapes[0],1,-1))
            else:
                outputs = tf.reshape(outputs, (1, shapes[0],-1))
            outputs_.append(outputs)
        outputs = sum(outputs_)
        if Params.bias:
            b = tf.get_variable("b", shape = outputs.shape[-1], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            outputs += b
        v = tf.get_variable("v", shape = outputs.shape[-1], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
        scores = tf.reduce_sum(tf.tanh(outputs) * v, [-1])
        if memory_len is not None:
            scores = mask_attn_score(scores, memory_len)
        return tf.nn.softmax(scores) # all attention output is softmaxed now

def cross_entropy_with_sequence_mask(output, target):
	cross_entropy = target * tf.log(output + 1e-8)
	cross_entropy = -tf.reduce_sum(cross_entropy, 2)
	mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
	cross_entropy *= mask
	cross_entropy = tf.reduce_sum(cross_entropy, 1)
	cross_entropy /= tf.reduce_sum(mask, 1)
	return tf.reduce_mean(cross_entropy)

def total_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total number of trainable parameters: {}".format(total_parameters))
