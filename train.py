# -*- coding: utf-8 -*-
# @Time    : 2019/9/12 15:42
# @Author  : PeterV
# @FileName: train.py
# @Software: PyCharm

import tensorflow as tf
from model import BiLSTM
from tqdm import tqdm
import os
from config import Config
import math
import logging
import random
from dataLoader import get_batch
from utils import calc_accuracy,get_parameter_list

index = int(random.random() * 10000)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

logging.info('# Config')
config = Config()
cfg = config.get_args()
#TODO Save config



logging.info('# Load model')
parameter_list = get_parameter_list(cfg)
for param in parameter_list:

	for fold in range(cfg.fold_num):

		model_output = '%s_fold%d_epoch%2dL_devAcc%.2f' % (cfg.version, fold, epoch, accuracy)
		parameter_string = 'layer_num_%d_cell_num_%d_dropout_%.2f' % (param['layer_num'],
																	  param['cell_num'],
																	  param['dropout'])
		ckpt_path = os.path.join(cfg.result_path, cfg.version, 'index{}_models'.format(str(index)), parameter_string,
								 str(fold))

		logging.info('#Preprocessing train/eval batches')
		train_batches, num_train_batches, num_train_samples = get_batch(cfg.data_npy_path, cfg.filename_x_train,
																		cfg.filename_y_train, cfg.epochs,
																		cfg.maxlen, cfg.len_wv, cfg.batch_size[0],
																		cfg.num_classes, str(fold),
																		shuffle=True)
		dev_batches, num_dev_batches, num_dev_samples = get_batch(cfg.data_npy_path, cfg.filename_x_dev,
																  cfg.filename_y_dev, cfg.epochs,
																  cfg.maxlen, cfg.len_wv, cfg.batch_size[0],
																  cfg.num_classes, str(fold),
																  shuffle=False)

		# create a iterator of the correct shape and type
		iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
		xs, ys = iter.get_next()

		train_init_opt = iter.make_initializer(train_batches)
		dev_init_opt = iter.make_initializer(dev_batches)

		# index+=1
		model = BiLSTM(param)
		# print('xs')
		# print(xs)
		# print('ys')
		# print(ys)
		loss,train_opt,pred_train,train_summaries,global_step,lstm_cell_fw,x_check = model.train(xs,ys)
		logits_eval,probs_eval,pred_eval,ys = model.eval(xs,ys)

		#Variables for early stop
		dev_history = []

		dev_best = 0

		stop_times = 0

		logging.info('# Session')
		saver = tf.train.Saver(max_to_keep=model.epoch)
		with tf.Session() as sess:
			ckpt = tf.train.latest_checkpoint(ckpt_path)
			if ckpt is None:
				logging.info("Initializing from scratch")
				sess.run(tf.global_variables_initializer())
				#TODO save_variable_speces
			else:
				saver.restore(sess,ckpt)

		# summary_writer = tf.summary.FileWriter(cfg.logdir,sess.graph)

			sess.run(train_init_opt)
			total_steps = param['epoch'] * num_train_batches
			_gs = sess.run(global_step)

			for i in tqdm(range(_gs,total_steps+1)):
				_,_gs,x_check_checking = sess.run([train_opt,global_step,x_check])
				# print('x_check')
				# print(x_check_checking[0][0])
				# _ = sess.run(train_opt)
				epoch = math.ceil(_gs / num_train_batches)

				if _gs and _gs % num_train_batches == 0:
					logging.info("epoch {} is done".format(epoch))
					_loss,lstm_cell = sess.run([loss,lstm_cell_fw]) #train loss


					logging.info("train loss{}".format(_loss))

					logging.info("# dev evaluation")
					sess.run(dev_init_opt)
					dev_results = []
					dev_labels = []
					cnt=0
					for _ in range(num_dev_batches):
						# cnt+=1
						# print(cnt)
						tmp_pred,tmp_target = sess.run([pred_eval,ys])
						dev_results.extend(tmp_pred)
						dev_labels.extend(tmp_target)
					# print('DEV3')
					# print(len(dev_results))
					# print(len(dev_labels))
					accuracy = calc_accuracy(dev_results,dev_labels)

					dev_history.append(accuracy)
					if accuracy > dev_best:
						dev_best = accuracy
						stop_times = 0
					else:
						stop_times += 1

					if stop_times > cfg.patience:
						logging.info('The model did not improve after{} times, you have got an excellent'
									 +'enough model.')
						break


					logging.info('# The dev accuracy is:{}'.format(accuracy))
					logging.info('# The best dev accuracy is{}'.format(dev_best))
					logging.info('# The times model does not improve is:{}'.format(stop_times))

					if stop_times == 0:
						logging.info('# save models')
						if not os.path.exists(ckpt_path):
							os.makedirs(ckpt_path)
						ckpt_name = os.path.join(ckpt_path,model_output)
						saver.save(sess,ckpt_name,global_step=global_step,write_meta_graph=False)
						logging.info("After training of {} epochs, {} has been saved".format(
							epoch,ckpt_name
						))

					logging.info('# fall back to train mode')
					sess.run(train_init_opt)

		del train_batches, num_train_batches, num_train_samples
		del dev_batches, num_dev_batches, num_dev_samples
		tf.reset_default_graph()		#reset computing graph for next parameter / fold data

	logging.info('Done one parameter')
logging.info('Done')

