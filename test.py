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

	X_max_seq_len = get_max_seq_len(X_data)

	# see results for some real translation case from training data
	# predict from model
	X_test_data = X_data[:100]
	X_test = prepare_input_data(data=X_test_data, max_seq_len=X_max_seq_len, unit_to_int=X_unit_to_int)

	# load model
	model = load_model(SAVED_MODEL_PATH)

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
