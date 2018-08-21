import os
from rnn import rnn_predict, read_input_file, write_output_to_file

DATA_LOCATION = os.getcwd()
INPUT_FILE = os.path.join(DATA_LOCATION, 'input_files', 'water_6_17_reduced_2_cap_rs.csv')


if __name__ == '__main__':
    predictor = rnn_predict(os.path.join(DATA_LOCATION, 'input_files', 'daily_rnn_retrained_16.pickle'), alpha=0.38, log=True)
    inputs, networks, targets = read_input_file(INPUT_FILE)
    predictions = predictor.predict(inputs, networks=networks, targets=targets, recal=True)
    write_output_to_file(targets, predictions, os.path.join(DATA_LOCATION, 'predictions_2017.csv'))
    predictor.save(os.path.join(DATA_LOCATION, 'input_files', 'daily_rnn_retrained_17.pickle'))