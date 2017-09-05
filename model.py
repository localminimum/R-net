# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

import tensorflow as tf
from tqdm import tqdm
from data_load import get_batch
from params import Params
from layers import *
from GRU import gated_attention_GRUCell
from evaluate import *
import numpy as np
import cPickle as pickle
from process import *

optimizer_factory = {"adadelta":tf.train.AdadeltaOptimizer,
					"adam":tf.train.AdamOptimizer,
					"gradientdescent":tf.train.GradientDescentOptimizer,
					"adagrad":tf.train.AdagradOptimizer}

class Model(object):
	def __init__(self,is_training = True):
		self.is_training = is_training
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			data, self.num_batch = get_batch(is_training = is_training)
			(self.passage_w,
			self.question_w,
			self.passage_c,
			self.question_c,
			self.passage_w_len,
			self.question_w_len,
			self.passage_c_len,
			self.question_c_len,
			self.indices) = data
			self.passage_w_len = tf.squeeze(self.passage_w_len)
			self.question_w_len = tf.squeeze(self.question_w_len)

			self.encode_ids()
			self.params = get_attn_params(Params.attn_size, initializer = tf.contrib.layers.xavier_initializer)
			self.attention_match_rnn()
			self.bidirectional_readout()
			self.pointer_network()

			if is_training:
				self.loss_function()
				self.summary()
				self.init_op = tf.global_variables_initializer()
			else:
				self.outputs()
			total_params()

	def encode_ids(self):
		with tf.device('/cpu:0'):
			self.char_embeddings = tf.Variable(tf.constant(0.0, shape=[Params.char_vocab_size, Params.emb_size]),trainable=False, name="char_embeddings")
			self.char_embeddings_placeholder = tf.placeholder(tf.float32,[Params.char_vocab_size, Params.emb_size],"char_embeddings_placeholder")
			self.word_embeddings = tf.Variable(tf.constant(0.0, shape=[Params.vocab_size, Params.emb_size]),trainable=False, name="word_embeddings")
			self.word_embeddings_placeholder = tf.placeholder(tf.float32,[Params.vocab_size, Params.emb_size],"word_embeddings_placeholder")
			self.emb_assign = tf.group(tf.assign(self.word_embeddings, self.word_embeddings_placeholder),tf.assign(self.char_embeddings, self.char_embeddings_placeholder))

		# Embed the question and passage information for word and character tokens
		self.passage_word_encoded, self.passage_char_encoded = encoding(self.passage_w,
														self.passage_c,
														word_embeddings = self.word_embeddings,
														char_embeddings = self.char_embeddings,
														scope = "passage_embeddings")
		self.question_word_encoded, self.question_char_encoded = encoding(self.question_w,
														self.question_c,
														word_embeddings = self.word_embeddings,
														char_embeddings = self.char_embeddings,
														scope = "question_embeddings")
		self.passage_char_encoded = bidirectional_GRU(self.passage_char_encoded,
														self.passage_c_len,
														scope = "passage_char_encoding",
														output = 1,
														is_training = self.is_training)
		self.question_char_encoded = bidirectional_GRU(self.question_char_encoded,
														self.question_c_len,
														scope = "question_char_encoding",
														output = 1,
														is_training = self.is_training)
		self.passage_encoding = tf.concat((self.passage_word_encoded, self.passage_char_encoded),axis = 2)
		self.question_encoding = tf.concat((self.question_word_encoded, self.question_char_encoded),axis = 2)

		# Passage and question encoding
		self.passage_encoding = bidirectional_GRU(self.passage_encoding,
													self.passage_w_len,
													layers = Params.num_layers,
													scope = "passage_encoding",
													output = 0,
													is_training = self.is_training)
		self.question_encoding = bidirectional_GRU(self.question_encoding,
													self.question_w_len,
													layers = Params.num_layers,
													scope = "question_encoding",
													output = 0,
													is_training = self.is_training)

	def attention_match_rnn(self):
		with tf.variable_scope("attention_match_rnn"):
			memory = self.question_encoding
			inputs = self.passage_encoding
			scopes = ["question_passage_matching", "self_matching"]
			params = [([[self.params["W_u_Q"],
						self.params["W_u_P"],
						self.params["W_v_P"]],
						self.params["v"]],self.params["W_g"]),
					([[tf.concat((self.params["W_v_P"],
						self.params["W_v_P"]),axis = 0),
						self.params["W_v_Phat"]],
						self.params["v"]],self.params["W_g"])]
			for i in range(2):
				if scopes[i] == "question_passage_matching":
					cell_fw = gated_attention_GRUCell(Params.attn_size, memory = memory, params = params[i], self_matching = False)
					cell_bw = gated_attention_GRUCell(Params.attn_size, memory = memory, params = params[i], self_matching = False)
				elif scopes[i] == "self_matching":
					cell_fw = gated_attention_GRUCell(Params.attn_size, memory = memory, params = params[i], self_matching = True)
					cell_bw = gated_attention_GRUCell(Params.attn_size, memory = memory, params = params[i], self_matching = True)
				cell = (cell_fw, cell_bw)
				inputs = attention_rnn(inputs,
										self.passage_w_len,
										Params.attn_size,
										cell,
										bidirection = True,
										scope = scopes[i])
				memory = inputs # self matching (attention over itself)
				inputs = apply_dropout(inputs, is_training = self.is_training)
			self.self_matching_output = inputs

	def bidirectional_readout(self):
		self.final_bidirectional_outputs = bidirectional_GRU(self.self_matching_output,
															self.passage_w_len,
															layers = Params.num_layers,
															scope = "bidirectional_readout",
															output = 0,
															is_training = self.is_training)

	def pointer_network(self):
		params = (([self.params["W_u_Q"],self.params["W_v_Q"]],self.params["v"]),
					([self.params["W_h_P"],self.params["W_h_a"]],self.params["v"]))
		cell = apply_dropout(tf.contrib.rnn.GRUCell(Params.attn_size*2), is_training = self.is_training)
		self.points_logits = pointer_net(self.final_bidirectional_outputs, self.passage_w_len, self.question_encoding, cell, params, scope = "pointer_network")

	def outputs(self):
		self.output_index = tf.argmax(self.points_logits, axis = 2)

	def loss_function(self):
		with tf.variable_scope("loss"):
			shapes = self.passage_w.shape
			self.mask = tf.to_float(tf.sequence_mask(self.passage_w_len, shapes[1]))
			self.points_logits *= tf.expand_dims(self.mask,1)

			# Causes NaN error
			# self.mean_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.indices, logits = self.points_logits))

			# Use non-sparse softmax
			self.indices_prob = tf.one_hot(self.indices, shapes[1])
			self.mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.indices_prob, logits = self.points_logits))
			self.optimizer = optimizer_factory[Params.optimizer](Params.opt_arg[Params.optimizer])

			if Params.clip:
				# gradient clipping by norm
				gradients, variables = zip(*self.optimizer.compute_gradients(self.mean_loss))
				gradients, _ = tf.clip_by_global_norm(gradients, Params.norm)
				self.train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step = self.global_step)
			else:
				self.train_op = self.optimizer.minimize(self.mean_loss, global_step = self.global_step)

	def summary(self):
		self.F1 = tf.Variable(tf.constant(0.0, shape=(), dtype = tf.float32),trainable=False, name="F1")
		self.F1_placeholder = tf.placeholder(tf.float32, shape = (), name = "F1_placeholder")
		self.EM = tf.Variable(tf.constant(0.0, shape=(), dtype = tf.float32),trainable=False, name="EM")
		self.EM_placeholder = tf.placeholder(tf.float32, shape = (), name = "EM_placeholder")
		self.metric_assign = tf.group(tf.assign(self.F1, self.F1_placeholder),tf.assign(self.EM, self.EM_placeholder))
		tf.summary.scalar('mean_loss', self.mean_loss)
		tf.summary.scalar("training_F1_Score",self.F1)
		tf.summary.scalar("training_Exact_Match",self.EM)
		tf.summary.scalar('learning_rate', Params.learning_rate)
		self.merged = tf.summary.merge_all()

def debug():
	model = Model(is_training = False)
	print("Built model")

def test():
	model = Model(is_training = False); print("Built model")
	dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))
	glove = np.memmap(Params.data_dir + "glove.np", dtype = np.float32, mode = "r")
	glove = np.reshape(glove,(Params.vocab_size,Params.emb_size))
	char_glove = np.memmap(Params.data_dir + "glove_char.np",dtype = np.float32, mode = "r")
	char_glove = np.reshape(char_glove,(Params.char_vocab_size,Params.emb_size))
	with model.graph.as_default():
		sv = tf.train.Supervisor()
		with sv.managed_session() as sess:
			sv.saver.restore(sess, tf.train.latest_checkpoint(Params.logdir))
			sess.run(model.emb_assign, {model.word_embeddings_placeholder:glove, model.char_embeddings_placeholder:char_glove})
			EM, F1 = 0.0, 0.0
			for step in tqdm(range(model.num_batch), total = model.num_batch, ncols=70, leave=False, unit='b'):
				index, ground_truth, passage = sess.run([model.output_index, model.indices, model.passage_w])
				for batch in range(Params.batch_size):
					f1, em = f1_and_EM(index[batch], ground_truth[batch], passage[batch], dict_)
					F1 += f1
					EM += em
			F1 /= float(model.num_batch * Params.batch_size)
			EM /= float(model.num_batch * Params.batch_size)
			print("Exact_match: {}\nF1_score: {}".format(EM,F1))

def main():
	model = Model(is_training = True); print("Built model")
	dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))
	glove = np.memmap(Params.data_dir + "glove.np", dtype = np.float32, mode = "r")
	glove = np.reshape(glove,(Params.vocab_size,300))
	char_glove = np.memmap(Params.data_dir + "glove_char.np",dtype = np.float32, mode = "r")
	char_glove = np.reshape(char_glove,(Params.char_vocab_size,300))
	with model.graph.as_default():
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sv = tf.train.Supervisor(logdir=Params.logdir,
								save_model_secs=0,
								global_step = model.global_step,
								init_op = model.init_op)
		with sv.managed_session(config = config) as sess:
			sess.run(model.emb_assign, {model.word_embeddings_placeholder:glove, model.char_embeddings_placeholder:char_glove})
			for epoch in range(1, Params.num_epochs+1):
				if sv.should_stop(): break
				for step in tqdm(range(model.num_batch), total = model.num_batch, ncols=70, leave=False, unit='b'):
					sess.run(model.train_op)
					if step % Params.save_steps == 0:
						sv.saver.save(sess, Params.logdir + '/model_epoch_%d_step_%d'%(epoch,step))
						index, ground_truth, passage = sess.run([model.points_logits, model.indices, model.passage_w])
						index = np.argmax(index, axis = 2)
						F1, EM = 0.0, 0.0
						for batch in range(Params.batch_size):
							f1, em = f1_and_EM(index[batch], ground_truth[batch], passage[batch], dict_)
							F1 += f1
							EM += em
						F1 /= float(Params.batch_size)
						EM /= float(Params.batch_size)
						sess.run(model.metric_assign,{model.F1_placeholder: F1, model.EM_placeholder: EM})
						print("\nExact_match: {}\nF1_score: {}".format(EM,F1))

if __name__ == '__main__':
	if Params.debug == True:
		print("Debugging...")
		debug()
	elif Params.test == True:
		print("Testing on dev set...")
		test()
	else:
		print("Training...")
		main()
