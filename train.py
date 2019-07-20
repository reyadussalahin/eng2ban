import tensorflow as tf

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import numpy as np

import os

from util import get_data, get_unit_int_conversion_helper, get_max_seq_len
from util import add_command_unit_to_output_data
from util import prepare_input_data, prepare_output_data
from util import create_model


from config import config


# configuration parameters
FILE_PATH = config['file_path']
NUM_OF_HIDDEN_UNIT = config['num_of_hidden_unit']
BATCH_SIZE = config['batch_size']
EPOCHS = config['epochs']
VALIDATION_SPLIT = config['validation_split']
SAVED_DIR = config['saved_dir']
SAVED_MODEL_PATH = config['saved_model_path']
SAVED_EPOCH_PATH = config['saved_epoch_path']
SAVED_ACCURACY_PATH = config['saved_accuracy_path']
SAVE_MODEL_AFTER_EPOCH = config['save_model_after_epoch']
TRANSLATION_OUTPUT_PATH = config['translation_output_path']


# main
if __name__ == '__main__':
	# split each line into input and output sentence. Then each input/output sentence into words
	# X_data 2D array of strings contains sentences of input data where each sentence is broken into words
	# like ['i', 'am', 'fine', '.']
	#      ['how', 'are', 'you', '?']
	#      ........
	#      ........
	# y_data is similiar to X_data but for output data
	X_data, y_data = get_data(FILE_PATH)

	# producing conversion data structures for input and output data.
	# 'GO', 'PAD', 'EOS', 'UNK' is added in conversion data structures
	X_unit_to_int, X_int_to_unit = get_unit_int_conversion_helper(X_data)
	y_unit_to_int, y_int_to_unit = get_unit_int_conversion_helper(y_data)

	# for k, v in X_unit_to_int.items():
	# 	print('{}: {}'.format(k, v))

	# for i, v in enumerate(y_int_to_unit):
	# 	print('{}: {}'.format(i, v))


	X_max_seq_len = get_max_seq_len(X_data)
	X = prepare_input_data(data=X_data, max_seq_len=X_max_seq_len, unit_to_int=X_unit_to_int)
	
	# adding 'GO' before starting and 'EOS' at end of sequence
	add_command_unit_to_output_data(y_data)
	y_max_seq_len = get_max_seq_len(y_data)
	y = prepare_output_data(data=y_data, max_seq_len=y_max_seq_len, unit_to_int=y_unit_to_int)


	# checking modified y_data
	# with open('output.txt', 'w', encoding='utf-8') as file:
	# 	i = 0
	# 	for seq in y_data:
	# 		b = ''
	# 		for unit in seq:
	# 			b += unit + ' '
	# 		file.write(b)
	# 		file.write('\n')
	# 		i += 1
	# 		if i == 10:
	# 			break


	# count = 0
	# for seq in X:
	# 	print(seq)
	# 	count += 1
	# 	if count == 20:
	# 		break

	# print(y_data)
	# count = 0
	# for frame in y:
	# 	buf = ''
	# 	for pos in frame:
	# 		placing = 0
	# 		for _int in pos:
	# 			if _int == 1:
	# 				buf += str(placing) + ' '
	# 				break;
	# 			placing += 1
	# 	count += 1
	# 	print(buf)
	# 	if count == 20:
	# 		break

	# count = 0
	# for frame in y:
	# 	print(frame)
	# 	count += 1
	# 	if count == 10:
	# 		break


	X_num_of_unit = len(X_unit_to_int)
	y_num_of_unit = len(y_unit_to_int)
	

	# print('x-num-unit: {}'.format(X_num_of_unit))
	# print('y-num-unit: {}'.format(y_num_of_unit))

	# print('x-max-seq-unit: {}'.format(X_max_seq_len))
	# print('y-max-seq-unit: {}'.format(y_max_seq_len))

	if not os.path.exists(SAVED_DIR):
		os.mkdir(SAVED_DIR)

	while EPOCHS > 0:
		epochs = min(EPOCHS, SAVE_MODEL_AFTER_EPOCH)
		# load model
		if os.path.isfile(SAVED_MODEL_PATH): # retrieving model
			model = load_model(SAVED_MODEL_PATH)
			with open(SAVED_EPOCH_PATH, 'r') as f:
				prev_epoch = int(f.read())
		else: # creating model
			model = create_model(X_num_of_unit, X_max_seq_len, y_num_of_unit, y_max_seq_len, NUM_OF_HIDDEN_UNIT)
			prev_epoch = 0

		# printing model summary
		# model.summary()

		# shuffle data
		# to avoid local minima
		indices = np.arange(len(X))
		np.random.shuffle(indices)
		X = X[indices]
		y = y[indices]


		# checkpoint = ModelCheckpoint(filepath=SAVED_MODEL_PATH, monitor='acc',
		# 	verbose=1, save_best_only='true', mode='max')
		# callback_list = [checkpoint]
		# start training
		# model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS,
		# 	validation_split=VALIDATION_SPLIT, callbacks=callback_list)
		
		# start training
		model.fit(X, y, batch_size=BATCH_SIZE, epochs=epochs, validation_split=VALIDATION_SPLIT)
		model.save(SAVED_MODEL_PATH)

		with open(SAVED_EPOCH_PATH, 'w') as f:
			now_epoch = prev_epoch + epochs
			f.write(str(now_epoch))

		EPOCHS = EPOCHS - epochs
		print('trained {} epoch till now...'.format(now_epoch))

		accuracy = model.evaluate(X, y)
		with open(SAVED_ACCURACY_PATH, 'w') as f:
			f.write(str(accuracy))
		print('accuracy on overall data after training {} epoch: {}'.format(now_epoch, accuracy))


	# see results for some real translation case from training data
	# predict from model
	X_test_data = X_data[2000:]
	X_test = prepare_input_data(data=X_test_data, max_seq_len=X_max_seq_len, unit_to_int=X_unit_to_int)

	print('predicting translations...')
	preds = np.argmax(model.predict(X_test), axis=2)

	# get sequences from predictions
	pred_seqs = []
	for pred in preds:
		# pred_seq = ' '.join(y_int_to_unit[idx] for idx in pred if idx > 0)
		pred_seq = ' '.join(y_int_to_unit[idx] for idx in pred)
		# print(pred_seq)
		pred_seqs.append(pred_seq)

	with open(TRANSLATION_OUTPUT_PATH, 'w', encoding='utf-8') as output_file:
		print('writing translated sentences in output file...')
		for i in range(len(pred_seqs)):
			output_file.write('{} >> {}'.format(X_test_data[i], pred_seqs[i]))
			output_file.write('\n')

