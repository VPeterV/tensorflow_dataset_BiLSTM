# -*- coding: utf-8 -*-
# @Time    : 2019/9/16 22:56
# @Author  : PeterV
# @FileName: dataLoader.py
# @Software: PyCharm

import tensorflow as tf
from utils import calc_num__batches
import numpy as np

def get_num_data_npy(data_path,file_path_y,index):
	npy_y = np.load(data_path+file_path_y+index+'.npy',allow_pickle=True)

	return len(npy_y)

def load_data_npy(data_path,file_path_x,file_path_y,index):
	npy_x = np.load(data_path+file_path_x+index+'.npy',allow_pickle=True)
	npy_y = np.load(data_path+file_path_y+index+'.npy',allow_pickle=True)

	return npy_x,npy_y

def padding(npy_x,maxlen):
	return tf.keras.preprocessing.sequence.pad_sequences(npy_x,maxlen,dtype='float32')

def target2onehot(y_npy,num_classes):
	y_onehot = []
	for item in y_npy:
		# print(item)
		tmp = [0] * num_classes
		tmp[int(item)] = 1
		y_onehot.append(tmp)
	y_onehot = np.array(y_onehot)
	# print(y_onehot)

	return y_onehot

def generator_fn(x_npy,y_npy):
	'''
	Generate training / evaluation data
	:param x_npy:
	:param y_npy:
	:return:
	'''
	for x,y in zip(x_npy,y_npy):
		yield x,y


def input_fn(x_npy,y_npy,batch_size,epoch,num_classes,maxlen,len_wv,shuffle=False):
	'''
	Batchify data
	:param x_npy:
	:param y_npy:
	:param batch_size:
	:param shuffle:
	:return:
	'''
	shapes = ([maxlen,len_wv]),([num_classes])
	types = (tf.float32,tf.float32)
	dataset = tf.data.Dataset.from_generator(
		generator_fn,
		output_shapes=shapes,
		output_types=types,
		args=(x_npy,y_npy)
	)
	if shuffle:
		dataset = dataset.shuffle(128*batch_size)

	dataset = dataset.repeat() #iterator forever == While True
	dataset = dataset.batch(batch_size)

	return dataset


def get_batch(data_path,file_path_x,file_path_y,epoch,maxlen,len_wv,batch_size,num_classes,index,shuffle=False):
	'''
	:param file_path_x: word vector x
	:param file_path_y: targets
	:param maxlen: maxlen
	:param batch_size: batch size
	:param shuffle: Train:True Eval or Test;False
	:return: batches,num_batches:number of mini-batches,num_samples
	'''
	x_npy,y_npy = load_data_npy(data_path,file_path_x,file_path_y,index)
	x_npy = padding(x_npy,maxlen)
	# print('x padding')
	# print(x_npy[0][-1])
	y_npy = target2onehot(y_npy,num_classes)
	batches = input_fn(x_npy,y_npy,batch_size,epoch,num_classes,maxlen,len_wv,shuffle=shuffle)
	assert len(x_npy)==len(y_npy)
	num_batches =calc_num__batches(len(x_npy),batch_size)
	return batches,num_batches,len(x_npy)
