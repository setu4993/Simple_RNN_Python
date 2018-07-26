import os
from rnn import *

DATA_LOCATION = 'D:\Programming\Visual Studio Projects\simple_rnn_seq_test_train_rec_3\simple_rnn_seq_test_train_rec_3'
WEIGHTS_LOCATION = os.path.join(DATA_LOCATION, '100k_8_hidden_daily_rec_022_parallel_jan_21_retrained_16.txt')
INPUT_FILE = os.path.join(DATA_LOCATION, 'water_6_17_reduced_2_cap_rs.csv')

wih, whh, who, prh = load_weights_from_pickle_dump(os.path.join(DATA_LOCATION, 'rnn_retrained_16.pickle'))
rnn = recurrent_neural_network([wih, whh, who, prh], alpha=0.38)

inputs, networks, targets = read_input_file(INPUT_FILE)

predictions = rnn.run_network(inputs, networks=networks, targets=targets, recal=True)

write_output_to_file(targets, predictions, os.path.join(DATA_LOCATION, 'voila.csv'))
