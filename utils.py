# -*- coding: utf-8 -*-
# @Time    : 2019/9/16 22:57
# @Author  : PeterV
# @FileName: utils.py
# @Software: PyCharm

import tensorflow as tf
import json
import os
import re
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)



def calc_num__batches(total_num,batch_size):
	'''Calculate the number of batches.
	:param total_num: total sample number
	:param batch_size:
	:return:
	'''
	return total_num // batch_size + int(total_num % batch_size !=0)

def calc_accuracy(y_preds,y_targets):
	# y_pred_array = np.array(y_pred)
	# y_target_array = np.array(y_target)
	# correct = [float(y == y_) for y,y_ in zip(y_pred_array,y_target_array)]
	# accuracy = sum(correct) / len(correct)
	print(len(y_preds))
	print(len(y_targets))
	assert len(y_preds) == len(y_targets)
	# print(y_preds)
	# print(y_targets)
	y_targets = list(np.argmax(np.array(y_targets),-1))
	correct_num = 0
	for y_pred,y_target in zip(y_preds,y_targets):
		if y_pred == y_target:
			correct_num+=1
	accuracy = correct_num/len(y_preds)
	return accuracy

def get_parameter_list(config):
	cfg = config
	parameter_list = []
	for layer_num in cfg.layer_nums:
		for cell_num in cfg.cell_nums:
			for hidden_nums in cfg.hidden_nums:
				for dropout in cfg.dropouts:
					for epoch in cfg.epochs:
						for batch_size in cfg.batch_size:
							for lr in cfg.lrs:
								param = dict()
								param['layer_num'] = layer_num
								param['cell_num'] = cell_num
								param['hidden_num'] = hidden_nums
								param['dropout'] = float(dropout)
								param['epoch'] = epoch
								param['batch_size'] = batch_size
								param['lr'] = lr
								param['maxlen'] = cfg.maxlen
								param['len_wv'] = cfg.len_wv
								param['num_classes'] = cfg.num_classes

								parameter_list.append(param)

	return parameter_list
