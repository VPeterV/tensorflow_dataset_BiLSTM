# -*- coding: utf-8 -*-
# @Time    : 2019/9/12 15:11
# @Author  : PeterV
# @FileName: model.py
# @Software: PyCharm

import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)

class BiLSTM:
	'''
	xs : tuple of
		x:float32 tensor (N,maxlen,vector_size)
	ys : tuple of
		y:int32 tensor (N,1)
	training : boolean
	'''
	def __init__(self,param):
		self.param = param
		self.cell_num = param['cell_num']
		self.layer_num = param['layer_num']
		self.maxlen = param['maxlen']
		self.batch_size = param['batch_size']
		self.dropout = param['dropout']
		self.lr = param['lr']
		self.epoch = param['epoch']
		self.num_classes = param['num_classes']

	def build(self,xs,training=True):
		'''
		:param xs:
		:param training:
		:return:
		'''
		with tf.variable_scope('builder',reuse=tf.AUTO_REUSE):
			x = xs
			# print(x)

			#No embedding layer since we have converted word 2 vector (Both w2v and Bert)
			lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.cell_num)
			lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.cell_num)

			# lstm_cell_fw_muti = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw]*self.layer_num)
			# lstm_cell_bw_muti = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw]*self.layer_num)

			outputs_seq,outputs_states = \
				tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw,lstm_cell_bw,x,
												dtype=tf.float32)
			final_state_fw,final_state_bw = outputs_states
			outputs = tf.reshape(tf.concat(values=[final_state_fw.h,final_state_bw.h],axis=1),[-1,self.cell_num*2]) #[batch_size,cell_num*2]
			outputs_dropout = tf.layers.dropout(outputs,rate=self.dropout,training=training)

			#Dense(FC) for softmax
			softmax_w = tf.get_variable('softmax_w',[self.cell_num*2,self.num_classes],initializer=tf.initializers.random_normal())
			softmax_b = tf.get_variable('softmax_b',[self.num_classes],initializer=tf.zeros_initializer())
			logits = tf.nn.bias_add(tf.matmul(outputs_dropout,softmax_w),softmax_b)
			# logits = tf.layers.dense(outputs_dropout,self.num_classes,activation='softmax')

		memory = logits

		return memory,lstm_cell_fw

	def train(self,xs,y):
		'''

		:param xs:
		:param ys:
		:return: loss:loss,train_opt:opt,pred:train prediction
		'''
		#forward
		logits,lstm_cell_fw = self.build(xs)

		#train scheme
		# print('len_logits')
		# print(logits.shape[0])
		# loss = -tf.reduce_mean(y * tf.log(logits))
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y))

		global_step = tf.train.get_or_create_global_step()

		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		train_opt = optimizer.minimize(loss,global_step=global_step)
		# train_opt = optimizer.minimize(loss)

		probs = tf.nn.softmax(logits,-1)
		pred = tf.argmax(probs,-1)

		tf.summary.scalar('loss',loss)
		# tf.summary.scalar('global step',)

		summaries = tf.summary.merge_all()

		return loss,train_opt,pred,summaries,global_step,lstm_cell_fw.weights,xs

	def eval(self,xs,y):
		'''

		:param xs:
		:param y:
		:return:
		'''
		logits,lstm_cell_fw = self.build(xs,training=False)

		logging.info('Inference graph is being built. Please be patient')
		probs = tf.nn.softmax(logits,-1)
		pred = tf.argmax(probs,-1)

		return logits,probs,pred,y

