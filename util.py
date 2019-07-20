from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Bidirectional, RepeatVector, TimeDistributed
from keras.optimizers import Adam

import numpy as np

import re

from config import config

LEARNING_RATE = config['learning_rate']

command_unit = {
	'PAD' : 0, # for padding
	'GO'  : 1, # for start command in decoder input
	'EOS' : 2, # for end command in decoder input and also in decoder output while predicting
	'UNK' : 3  # for unknown words
}


def get_data(file_path): # returns 2D array of strings, each sentence is splitted into words
	X_data = []
	y_data = []
	with open(file_path, 'r', encoding='utf-8') as file:
		lines = file.read().split('\n')
		for line in lines:
			input_text, output_text = line.split('\t')
			X_data.append(re.findall(r"[\w']+|[.,!?;]", input_text))
			y_data.append(re.findall(r"[\u0980-\u09F6']+|[,!?;।]", output_text))

	return X_data, y_data


def get_unit_int_conversion_helper(data): # return unit_to_int and int_to_unit converter
	vocub = set()

	for seq in data:
		for unit in seq:
			if unit not in vocub:
				vocub.add(unit)

	# adding command units
	int_to_unit = ['PAD', 'GO', 'EOS', 'UNK']

	# adding units from vocub set
	int_to_unit.extend(list(vocub))
	
	# unit to int conversion
	unit_to_int = {unit:idx for idx, unit in enumerate(int_to_unit)}
	
	return unit_to_int, int_to_unit


def add_command_unit_to_output_data(data):
	for seq in data:
		seq.insert(0, 'GO')
		seq.append('EOS')


def get_max_seq_len(data):
	return max([len(seq) for seq in data])


def prepare_input_data(data, max_seq_len, unit_to_int):
	# prepares input_data from X_data
	# converts each unit to int
	# and also does padding
	# note that, we are reversing input data
	# because, it has been observed from experience that
	# it gives better result
	# example: ['i', 'am', 'fine', '.'] would be like
	# ['PAD', 'PAD', 'PAD', 'PAD', '.', 'fine', 'am', 'i']
	# and strings i.e. units would be converted to
	# integer with help of unit_to_int
	X = np.zeros((len(data), max_seq_len), dtype='float32')
	for i, seq in enumerate(data):
		for j, unit in enumerate(seq):
			if unit in unit_to_int:
				X[i][max_seq_len - j - 1] = unit_to_int[unit]
			else:
				X[i][max_seq_len - j - 1] = unit_to_int['UNK']

	return X


# def prepare_decoder_input_data(data, max_seq_len, unit_to_int):
# 	y = np.zeros((len(data), max_seq_len), dtype='float32')
# 	for i, seq in enumerate(data):
# 		for j, unit in enumerate(seq):
# 			if unit in unit_to_int:
# 				y[i][j] = unit_to_int[unit]
# 			else:
# 				y[i][j] = unit_to_int['UNK']
# 	return y


def prepare_output_data(data, max_seq_len, unit_to_int):
	# creating one hot vectorization of output data
	# it's a 3D matrix created using len(data), max_seq_len, len(unit_to_int) ie. length of vocub
	# if we take a single line that we may say that, the line is represented as below
	# unit pos vs int value of unit which is marked as 1, others are 0
	# in this example, unit_to_int['i'] = 3, unit_to_int['fine'] = 4, unit_to_int['am'] = max_seq_len-1
	# unit value (from 0 to max)
	# ^
	# |                i                am                 fine
	# |           _________________________________________________
	# |                0                0                  0
	# |                0                0                  0
	# |                0                0                  0
	# |                1                0                  0
	# |                0                0                  1
	# |                .                .                  .
	# |                .                .                  .
	# |                .                .                  .
	# |                .                .                  .
	# |                .                .                  .
	# |                0                0                  0
	# |                0                0                  0
	# |                0                1                  0
	# ------------------------------------------------------------->>> pos of unit in sequence
	# this is applied for each sequence
	y = np.zeros((len(data), max_seq_len, len(unit_to_int)), dtype='float32')
	for i, seq in enumerate(data):
		for j, unit in enumerate(seq):
			if unit in unit_to_int:
				y[i][j][unit_to_int[unit]] = 1
			else:
				y[i][j][unit_to_int['UNK']] = 1

	return y


def create_model(X_num_of_unit, X_max_seq_len, y_num_of_unit, y_max_seq_len, num_of_hidden_unit):
	model = Sequential()

	# building encoder
	model.add(Embedding(X_num_of_unit, num_of_hidden_unit, input_length=X_max_seq_len, mask_zero=True))
	model.add(Bidirectional(LSTM(num_of_hidden_unit)))
	# model.add(LSTM(num_of_hidden_unit))
	model.add(RepeatVector(y_max_seq_len))

	# building decoder
	model.add(Bidirectional(LSTM(num_of_hidden_unit, return_sequences=True)))
	# model.add(LSTM(num_of_hidden_unit, return_sequences=True))
	model.add(TimeDistributed(Dense(y_num_of_unit, activation='softmax')))

	# compiling model
	model.compile(optimizer=Adam(LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

	return model

