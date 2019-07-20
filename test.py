import numpy as np
from keras.models import load_model

from util import get_data, get_unit_int_conversion_helper, get_max_seq_len
from util import add_command_unit_to_output_data
from util import prepare_input_data, prepare_output_data
from util import create_model


from config import config


# configuration parameters
FILE_PATH = config['file_path']
SAVED_MODEL_PATH = config['saved_model_path']

if __name__ == '__main__':
	X_data, y_data = get_data(FILE_PATH)

	X_unit_to_int, X_int_to_unit = get_unit_int_conversion_helper(X_data)
	y_unit_to_int, y_int_to_unit = get_unit_int_conversion_helper(y_data)
	# for k, v in X_unit_to_int.items():
	# 	print('{}: {}'.format(k, v))

	# for i, v in enumerate(y_int_to_unit):
	# 	print('{}: {}'.format(i, v))

	X_max_seq_len = get_max_seq_len(X_data)
	X = prepare_input_data(data=X_data, max_seq_len=X_max_seq_len, unit_to_int=X_unit_to_int)

	# defining X_test
	X_test = X[0:1000]
	# print(X_test)

	# load model
	model = load_model(SAVED_MODEL_PATH)

	# get predictions
	preds = np.argmax(model.predict(X_test), axis=2)

	# get sequences from predictions
	seqs = []
	for pred in preds:
		seq = ' '.join(y_int_to_unit[idx] for idx in pred if idx > 0)
		# print(seq)
		seqs.append(seq)

	for i in range(len(seqs)):
		print('{} >> {}'.format(X_data[i], seqs[i]))
