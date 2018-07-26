import os
from rnn import *

DATA_LOCATION = os.getcwd()
INPUT_FILE = os.path.join(DATA_LOCATION, 'input_files', 'water_6_17_reduced_2_cap_rs.csv')


if __name__ == '__main__':
    rnn = recurrent_neural_network(os.path.join(DATA_LOCATION, 'input_files', 'rnn_retrained_16.pickle'), alpha=0.38, log=True)
    inputs, networks, targets = read_input_file(INPUT_FILE)
    predictions = rnn.run_network(inputs, networks=networks, targets=targets, recal=True)
    write_output_to_file(targets, predictions, os.path.join(DATA_LOCATION, 'input_files', 'predictions_2017.csv'))
