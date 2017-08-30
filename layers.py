# -*- coding: utf-8 -*-
#/usr/bin/python2

from tensorflow.contrib.rnn import MultiRNNCell
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

def get_attn_params(attn_size,initializer = tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope("attention_weights"):
        params = {"W_u_Q":tf.get_variable("W_u_Q",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer),
                "W_u_P":tf.get_variable("W_u_P",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer),
                "W_v_P":tf.get_variable("W_v_P",dtype = tf.float32, shape = (attn_size, attn_size), initializer = initializer),
                "W_g":tf.get_variable("W_g",dtype = tf.float32, shape = (4 * attn_size, 4 * attn_size), initializer = initializer),
                "W_h_P":tf.get_variable("W_h_P",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer),
                "W_v_Phat":tf.get_variable("W_v_Phat",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer),
                "W_h_a":tf.get_variable("W_h_a",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer),
                "W_v_Q":tf.get_variable("W_v_Q",dtype = tf.float32, shape = (attn_size, attn_size), initializer = initializer),
                "v":tf.get_variable("v", dtype = tf.float32, shape = attn_size, initializer = initializer)}
        return params

def encoding(word, char, word_embeddings, char_embeddings, scope = "embedding"):
    with tf.variable_scope(scope):
        word_encoding = tf.nn.embedding_lookup(word_embeddings, word)
        char_encoding = tf.nn.embedding_lookup(char_embeddings, char)
        return word_encoding, char_encoding

def apply_dropout(cell,dropout = 0.2, is_training = True):
    if is_training:
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
    else:
        return cell

def bidirectional_GRU(inputs, inputs_len, cell = None, units = 75, layers = 1, scope = "Bidirectional_GRU", output = 0, is_training = True, reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        if cell is not None:
            (cell_fw, cell_bw) = cell
        else:
            if layers > 1:
                cell_fw = MultiRNNCell([apply_dropout(tf.contrib.rnn.GRUCell(units),is_training = is_training) for _ in range(layers)])
                cell_bw = MultiRNNCell([apply_dropout(tf.contrib.rnn.GRUCell(units),is_training = is_training) for _ in range(layers)])
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

def pointer_net(passage, passage_len, question, cell, params, scope = "pointer_network"):
    with tf.variable_scope(scope):
        weights_q, weights_p = params
        shapes = passage.get_shape().as_list()
        initial_state = question_pooling(question, units = Params.attn_size, weights = weights_q, scope = "question_pooling")
        stacked_state = tf.stack([initial_state] * shapes[1],1)
        inputs = tf.reshape(tf.concat((passage, stacked_state),axis = 2),(shapes[0]*shapes[1],-1))
        scores = attention(inputs, Params.attn_size, weights_p, output_fn = None, scope = "attention")
        scores = tf.reshape(scores,(shapes[0],shapes[1],1))
        p1_logits = tf.squeeze(scores,-1)
        scores = tf.nn.softmax(scores)
        attention_pool = tf.reduce_sum(scores * passage,1)
        _, state = cell(attention_pool, initial_state)
        stacked_state = tf.stack([state] * shapes[1],1)
        inputs = tf.reshape(tf.concat((passage, stacked_state),axis = 2),(shapes[0]*shapes[1],-1))
        scores = attention(inputs, Params.attn_size, weights_p, output_fn = None, scope = "attention", reuse = True)
        p2_logits = tf.reshape(scores,(shapes[0],shapes[1]))
        return tf.stack((p1_logits,p2_logits),1)

def attention_rnn(inputs, inputs_len, units, attn_cell, scope = "gated_attention_rnn", is_training = True):
    with tf.variable_scope(scope):
        (cell_fw, cell_bw) = attn_cell
        outputs = bidirectional_GRU(inputs,
                                    inputs_len,
                                    cell = (cell_fw,cell_bw),
                                    scope = scope + "_bidirectional",
                                    output = 0,
                                    is_training = is_training)
        return outputs

def question_pooling(memory, units, weights, scope = "question_pooling"):
    with tf.variable_scope(scope):
        shapes = memory.get_shape().as_list()
        V_r = tf.get_variable("question_param", shape = units, dtype = tf.float32)
        V_r = tf.stack([V_r] * shapes[1], axis = 0)
        V_r = tf.stack([V_r] * Params.batch_size, axis = 0)
        inputs = tf.concat((memory, V_r), axis = 2)
        inputs = tf.reshape(inputs, (shapes[0]*shapes[1],-1))
        attn = attention(inputs, units, weights, scope = "question_attention_pooling")
        attn = tf.reshape(attn,(shapes[0],shapes[1],1))
        return tf.reduce_sum(attn * memory, 1)

def gated_attention(memory, inputs, states, units, params, self_matching = False, output_argmax = None, scope="gated_attention"):
    with tf.variable_scope(scope):
        weights, W_g = params
        shapes = memory.get_shape().as_list()
        inputs_ = tf.stack([inputs] * shapes[1],1)
        inputs_ = tf.concat((memory,inputs_),axis = 2) # The order doesn't matter
        states = tf.reshape(states,(Params.batch_size,Params.attn_size))

        if not self_matching:
            states = tf.stack([states]*shapes[1],1)
            inputs_ = tf.concat((inputs_,states),axis = 2) # The order matters

        inputs_ = tf.reshape(inputs_,(shapes[0]*shapes[1],-1))
        scores = attention(inputs_, units, weights)
        scores = tf.reshape(scores,(shapes[0],shapes[1],1))
        attention_pool = tf.reduce_sum(scores * memory, 1)
        inputs = tf.concat((inputs,attention_pool),axis = 1)
        g_t = tf.sigmoid(tf.matmul(inputs,W_g))
        return g_t * inputs

def attention(inputs, units, weights, scope = "attention", output_fn = "softmax", reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        weights, v = weights
        shapes = inputs.get_shape().as_list()
        outputs = tf.matmul(inputs, weights)
        scores = tf.reduce_sum(tf.tanh(outputs) * v, [-1])
        if output_fn == "softmax":
            return tf.nn.softmax(scores)
        else:
            return scores

def total_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total number of trainable parameters: {}".format(total_parameters))
