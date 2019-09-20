# -*- coding: utf-8 -*-
# @Time    : 2019/9/19 15:04
# @Author  : PeterV
# @FileName: test.py
# @Software: PyCharm

import os
import tensorflow as tf
import numpy as np
import pandas as pd
from model import BiLSTM
from dataLoader import get_batch,get_num_data_npy
from config import Config
import logging
from utils import calc_accuracy,get_parameter_list
index = 5427 #The index of model

def evaluate_data(evaluation_type, test_labels,parameter_string,fold,all_probs,sess,init_opt,ys):
	logging.info('# {} evaluation'.format(evaluation_type))
	test_probs = []
	test_results = []
	for _ in range(num_test_bathces):
		tmp_probs, tmp_results, tmp_labels = sess.run([probs_test, pred_test, ys])
		test_probs.extend(tmp_probs)
		test_results.extend(tmp_results)
		test_labels.extend(tmp_labels)

	accuracy = calc_accuracy(test_results, test_labels)

	logging.info('The {} accuracy of parameter {} fold {} is {}'.format(evaluation_type,
																		parameter_string, fold, accuracy))

	if evaluation_type == 'Test':
		if fold == 0:
			all_probs = np.array(test_probs)
		else:
			all_probs += np.array(test_probs)

	logging.info('Reset the test iteration')
	if fold != cfg.fold_num - 1:
		test_labels = []

	if evaluation_type == 'Dev':
		sess.run(init_opt)
	else:
		pass
	return all_probs,test_labels,accuracy

if __name__ == '__main__':
	log_path =os.path.join('log',str(index))
	if not os.path.exists(log_path):
		os.makedirs(log_path)
	logging.getLogger().setLevel(logging.INFO)
	logging.basicConfig(level=logging.INFO, filename=os.path.join(log_path,'test_log.txt'), filemode='w',
						format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

	logging.info('# Config')
	config = Config()
	cfg = config.get_args()
	parameter_list = get_parameter_list(cfg)

	result_file = open(os.path.join(cfg.result_path,cfg.version,str(index)+'acc_result.csv'),'a',encoding='utf-8')
	result_file.write('layer_num,cell_num,hidden_num,num_classes,epoch,dropout,lr,batch_size,dev_acc,integrated'+'\n')

	test_batch_size = get_num_data_npy(cfg.data_npy_path, cfg.filename_y_test, '')
	dev_batch_sizes = [get_num_data_npy(cfg.data_npy_path, cfg.filename_y_dev, str(fold)) for fold in range(cfg.fold_num)]
	for param in parameter_list:
		all_probs_test = []
		all_probs_dev = []
		test_labels = []
		dev_labels = []
		accuracy_dev_all = []
		parameter_string = 'layer_num_%d_cell_num_%d_dropout_%.2f' % (param['layer_num'],
																	  param['cell_num'],
																	  param['dropout'])
		for fold in range(cfg.fold_num):

			logging.info('Preparing data')
			#change batch to dev sample nums
			test_batches, num_test_bathces, num_test_samples = get_batch(cfg.data_npy_path, cfg.filename_x_test,
																		 cfg.filename_y_test, cfg.epochs,
																		 cfg.maxlen, cfg.len_wv, test_batch_size,
																		 cfg.num_classes, index='', shuffle=False)

			#change batch to dev sample nums
			dev_batches, num_dev_batches, num_dev_samples = get_batch(cfg.data_npy_path,cfg.filename_x_dev,
																	  cfg.filename_y_dev,cfg.epochs,
																	  cfg.maxlen,cfg.len_wv,dev_batch_sizes[fold],
																	  cfg.num_classes,index=str(fold),shuffle=False)

			# create a iterator of the correct shape and type
			iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
			xs, ys = iter.get_next()

			test_init_opt = iter.make_initializer(test_batches)
			dev_init_opt = iter.make_initializer(dev_batches)

			logging.info('# load Model')
			model = BiLSTM(param)
			logits_test,probs_test,pred_test,ys = model.eval(xs,ys)

			logits_dev, probs_dev, pred_dev, ys = model.eval(xs, ys)

			logging.info('# Session')
			with tf.Session() as sess:
				model_path = os.path.join(cfg.result_path,cfg.version,'index'+str(index)+'_models',parameter_string,str(fold))
				ckpt = tf.train.latest_checkpoint(model_path)
				saver = tf.train.Saver()
				saver.restore(sess,ckpt)

				sess.run(dev_init_opt)  # begin iterate dev data
				all_probs_dev,dev_labels, accuracy_dev = evaluate_data('Dev',dev_labels,parameter_string,fold,all_probs_dev,sess,test_init_opt,ys) #Dev
				all_probs_test,test_labels, accuracy_test = evaluate_data('Test',test_labels,parameter_string,fold,all_probs_test,sess,dev_init_opt,ys) #Test

				accuracy_dev_all.append(accuracy_dev)


			tf.reset_default_graph()
		logging.info('Parameter {} is done'.format(parameter_string))

		logging.info('Dev average accuracy{}'.format(np.mean(accuracy_dev_all)))

		all_results_test = np.argmax(all_probs_test, -1)
		all_accuracy_test = calc_accuracy(all_results_test,test_labels)
		logging.info('Parameter {} is done'.format(parameter_string))
		logging.info('Test accuracy{}'.format(all_accuracy_test))
		result_file.write(str(param['layer_num'])+','+str(param['cell_num'])+','+str(param['hidden_num'])+','+str(param['num_classes'])
							+','+str(param['epoch'])+','+str(param['dropout'])+','+str(param['lr'])+','+str(cfg.batch_size[0])+','
								+str(np.mean(accuracy_dev_all))+','+str(all_accuracy_test)+'\n')






	logging.info('Test done!!!')



