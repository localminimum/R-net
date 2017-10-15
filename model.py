# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

import tensorflow as tf
from tqdm import tqdm
from data_load import get_batch, get_dev
from params import Params
from layers import *
from GRU import gated_attention_Wrapper, GRUCell, SRUCell
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
		# Build the computational graph when initializing
		self.is_training = is_training
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			self.data, self.num_batch = get_batch(is_training = is_training)
			(self.passage_w,
			self.question_w,
			self.passage_c,
			self.question_c,
			self.passage_w_len_,
			self.question_w_len_,
			self.passage_c_len,
			self.question_c_len,
			self.indices) = self.data

			self.passage_w_len = tf.squeeze(self.passage_w_len_)
			self.question_w_len = tf.squeeze(self.question_w_len_)

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
			self.char_embeddings = tf.Variable(tf.constant(0.0, shape=[Params.char_vocab_size, Params.char_emb_size]),trainable=True, name="char_embeddings")
			self.word_embeddings = tf.Variable(tf.constant(0.0, shape=[Params.vocab_size, Params.emb_size]),trainable=False, name="word_embeddings")
			self.word_embeddings_placeholder = tf.placeholder(tf.float32,[Params.vocab_size, Params.emb_size],"word_embeddings_placeholder")
			self.emb_assign = tf.assign(self.word_embeddings, self.word_embeddings_placeholder)

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
								cell_fn = SRUCell if Params.SRU else GRUCell,
								scope = "passage_char_encoding",
								output = 1,
								is_training = self.is_training)
		self.question_char_encoded = bidirectional_GRU(self.question_char_encoded,
								self.question_c_len,
								cell_fn = SRUCell if Params.SRU else GRUCell,
								scope = "question_char_encoding",
								output = 1,
								is_training = self.is_training)
		self.passage_encoding = tf.concat((self.passage_word_encoded, self.passage_char_encoded),axis = 2)
		self.question_encoding = tf.concat((self.question_word_encoded, self.question_char_encoded),axis = 2)

		# Passage and question encoding
		#cell = [MultiRNNCell([GRUCell(Params.attn_size, is_training = self.is_training) for _ in range(3)]) for _ in range(2)]
		self.passage_encoding = bidirectional_GRU(self.passage_encoding,
								self.passage_w_len,
								cell_fn = SRUCell if Params.SRU else GRUCell,
								layers = Params.num_layers,
								scope = "passage_encoding",
								output = 0,
								is_training = self.is_training)
		#cell = [MultiRNNCell([GRUCell(Params.attn_size, is_training = self.is_training) for _ in range(3)]) for _ in range(2)]
		self.question_encoding = bidirectional_GRU(self.question_encoding,
								self.question_w_len,
								cell_fn = SRUCell if Params.SRU else GRUCell,
								layers = Params.num_layers,
								scope = "question_encoding",
								output = 0,
								is_training = self.is_training)

	def attention_match_rnn(self):
		# Apply gated attention recurrent network for both query-passage matching and self matching networks
		with tf.variable_scope("attention_match_rnn"):
			memory = self.question_encoding
			inputs = self.passage_encoding
			scopes = ["question_passage_matching", "self_matching"]
			params = [(([self.params["W_u_Q"],
					self.params["W_u_P"],
					self.params["W_v_P"]],self.params["v"]),
					self.params["W_g"]),
				(([self.params["W_v_P_2"],
					self.params["W_v_Phat"]],self.params["v"]),
					self.params["W_g"])]
			for i in range(2):
				args = {"num_units": Params.attn_size,
						"memory": memory,
						"params": params[i],
						"self_matching": False if i == 0 else True,
						"memory_len": self.question_w_len if i == 0 else self.passage_w_len,
						"is_training": self.is_training,
						"use_SRU": Params.SRU}
				cell = [apply_dropout(gated_attention_Wrapper(**args), size = inputs.shape[-1], is_training = self.is_training) for _ in range(2)]
				inputs = attention_rnn(inputs,
							self.passage_w_len,
							Params.attn_size,
							cell,
							scope = scopes[i])
				memory = inputs # self matching (attention over itself)
			self.self_matching_output = inputs

	def bidirectional_readout(self):
		self.final_bidirectional_outputs = bidirectional_GRU(self.self_matching_output,
									self.passage_w_len,
									cell_fn = SRUCell if Params.SRU else GRUCell,
									# layers = Params.num_layers, # or 1? not specified in the original paper
									scope = "bidirectional_readout",
									output = 0,
									is_training = self.is_training)

	def pointer_network(self):
		params = (([self.params["W_u_Q"],self.params["W_v_Q"]],self.params["v"]),
				([self.params["W_h_P"],self.params["W_h_a"]],self.params["v"]))
		cell = apply_dropout(SRUCell(Params.attn_size*2), size = self.final_bidirectional_outputs.shape[-1], is_training = self.is_training)
		self.points_logits = pointer_net(self.final_bidirectional_outputs, self.passage_w_len, self.question_encoding, self.question_w_len, cell, params, scope = "pointer_network")

	def outputs(self):
		self.output_index = tf.argmax(self.points_logits, axis = 2)

	def loss_function(self):
		with tf.variable_scope("loss"):
			shapes = self.passage_w.shape
			self.indices_prob = tf.one_hot(self.indices, shapes[1])
			self.mean_loss = cross_entropy(self.points_logits, self.indices_prob)
			self.optimizer = optimizer_factory[Params.optimizer](**Params.opt_arg[Params.optimizer])

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
		self.dev_loss = tf.Variable(tf.constant(5.0, shape=(), dtype = tf.float32),trainable=False, name="dev_loss")
		self.dev_loss_placeholder = tf.placeholder(tf.float32, shape = (), name = "dev_loss")
		self.metric_assign = tf.group(tf.assign(self.F1, self.F1_placeholder),tf.assign(self.EM, self.EM_placeholder),tf.assign(self.dev_loss, self.dev_loss_placeholder))
		tf.summary.scalar('loss_training', self.mean_loss)
		tf.summary.scalar('loss_dev', self.dev_loss)
		tf.summary.scalar("F1_Score",self.F1)
		tf.summary.scalar("Exact_Match",self.EM)
		tf.summary.scalar('learning_rate', Params.opt_arg[Params.optimizer]['learning_rate'])
		self.merged = tf.summary.merge_all()

def debug():
	model = Model(is_training = False)
	print("Built model")

def test():
	model = Model(is_training = False); print("Built model")
	dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))
	with model.graph.as_default():
		sv = tf.train.Supervisor()
		with sv.managed_session() as sess:
			sv.saver.restore(sess, tf.train.latest_checkpoint(Params.logdir))
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
	init = False
	devdata, dev_ind = get_dev()
	if not os.path.isfile(os.path.join(Params.logdir,"checkpoint")):
		init = True
		glove = np.memmap(Params.data_dir + "glove.np", dtype = np.float32, mode = "r")
		glove = np.reshape(glove,(Params.vocab_size,Params.emb_size))
	with model.graph.as_default():
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sv = tf.train.Supervisor(logdir=Params.logdir,
						save_model_secs=0,
						global_step = model.global_step,
						init_op = model.init_op)
		with sv.managed_session(config = config) as sess:
			if init: sess.run(model.emb_assign, {model.word_embeddings_placeholder:glove})
			for epoch in range(1, Params.num_epochs+1):
				if sv.should_stop(): break
				for step in tqdm(range(model.num_batch), total = model.num_batch, ncols=70, leave=False, unit='b'):
					sess.run(model.train_op)
					if step % Params.save_steps == 0:
						gs = sess.run(model.global_step)
						sv.saver.save(sess, Params.logdir + '/model_epoch_%d_step_%d'%(gs//model.num_batch, gs%model.num_batch))
						sample = np.random.choice(dev_ind, Params.batch_size)
						feed_dict = {data: devdata[i][sample] for i,data in enumerate(model.data)}
						logits, dev_loss = sess.run([model.points_logits, model.mean_loss], feed_dict = feed_dict)
						index = np.argmax(logits, axis = 2)
						F1, EM = 0.0, 0.0
						for batch in range(Params.batch_size):
							f1, em = f1_and_EM(index[batch], devdata[8][sample][batch], devdata[0][sample][batch], dict_)
							F1 += f1
							EM += em
						F1 /= float(Params.batch_size)
						EM /= float(Params.batch_size)
						sess.run(model.metric_assign,{model.F1_placeholder: F1, model.EM_placeholder: EM, model.dev_loss_placeholder: dev_loss})
						print("\nDev_loss: {}\nDev_Exact_match: {}\nDev_F1_score: {}".format(dev_loss,EM,F1))

if __name__ == '__main__':
	if Params.mode.lower() == "debug":
		print("Debugging...")
		debug()
	elif Params.mode.lower() == "test":
		print("Testing on dev set...")
		test()
	elif Params.mode.lower() == "train":
		print("Training...")
		main()
	else:
		print("Invalid mode.")
