import csv
import pickle
import logging
import numpy as np

# Default values for variables. Ignored if weights are loaded from the weights pickle file.

INPUT_NODES = 3
HIDDEN_NODES = 8
OUTPUT_NODES = 1
TOTAL_NETWORKS = 12


def load_weights_from_csv(weights_csv_location):
    """
    Loads the weights from a CSV file into 4 objects. Not recommended for regular use. The CSV structure is complicated
    and prone to errors. Instead use `np.ndarray`s of equivalent type.

    :param weights_csv_location: Location of the weights CSV file on disk.
    :return: wih, whh, who, prh
    """
    wih = np.zeros((TOTAL_NETWORKS, INPUT_NODES + 1, HIDDEN_NODES))
    whh = np.zeros((TOTAL_NETWORKS, HIDDEN_NODES, HIDDEN_NODES))
    who = np.zeros((TOTAL_NETWORKS, HIDDEN_NODES + 1, OUTPUT_NODES))
    prh = np.zeros((TOTAL_NETWORKS, HIDDEN_NODES))

    weights_file = open(weights_csv_location, 'r')
    weights_data = csv.reader(weights_file)

    network = 0
    in_node = 0
    hi_node = 0
    ph_node = 0
    ho_node = 0

    for row in weights_data:
        if len(row) == 9:
            network = int(row[0])
            in_node = 0
            hi_node = 0
            ph_node = 0
            ho_node = 0
            shift = 1
        else:
            shift = 0
        if len(row) > 7:
            if in_node < INPUT_NODES + 1:
                for i, elem in enumerate(row[shift:]):
                    wih[network][in_node][i] = float(elem)
                in_node += 1
            else:
                for i, elem in enumerate(row):
                    whh[network][hi_node][i] = float(elem)
                hi_node += 1
        elif len(row) == 1:
            if ho_node < HIDDEN_NODES + 1:
                who[network][ho_node] = float(row[0])
                ho_node += 1
            else:
                prh[network][ph_node] = float(row[0])
                ph_node += 1
        else:
            raise FileNotFoundError('Incorrect format of input file')

        weights_file.close()
        return wih, whh, who, prh


def load_weights_from_pickle_dump(dump_location):
    """
    Returns the un-pickled output from a pickle-d array dump. Recommended for persisting the network state for next run.

    :param dump_location
    :return: [wih, whh, who, prh]
    """
    dump_file = open(dump_location, 'rb')
    list_weights = pickle.load(dump_file)
    dump_file.close()
    return list_weights


def save_weights_to_pickle_dump(dump_location, wih, whh, who, prh):
    """
    Stores a pickle-d array for persisting the network state for next run.

    :param dump_location, wih, whh, who, prh
    """
    dump_file = open(dump_location, 'wb')
    pickle.dump([wih, whh, who, prh], dump_file)
    dump_file.close()


def read_input_file(input_location):
    """

    :param input_location: Location on disk to the input CSV file for prediction.
    :return: inputs, networks, targets
    """
    input_file = open(input_location, 'r')
    input_lines = csv.reader(input_file)
    inputs = []
    networks = []
    targets = []
    for line in input_lines:
        inputs.append([int(line[0]), int(line[1]), int(line[2])])
        targets.append(float(line[5]))
        networks.append(int(line[6]) - 1)

    input_file.close()
    return inputs, networks, targets


def write_output_to_file(targets, predictions, output_location):
    """
    Writes the output to a CSV file.

    :param targets, predictions
    :param output_location: String containing the address on disk for the output file.
    """
    logging.info('Writing predictions to file')
    output_file = open(output_location, 'w', newline='')
    output_writer = csv.DictWriter(output_file, fieldnames=['Target', 'Prediction', 'Error', 'Error Percentage'])
    output_writer.writeheader()
    for t, p in zip(targets, predictions):
        output_writer.writerow({'Target': t, 'Prediction': p, 'Error': abs(t - p), 'Error Percentage':
            (abs(t - p) / t * 100)})
    output_file.close()
    logging.info('Predictions written to file')


def log_to_console():
    """
    Initiates a console logger. Recommended if running through an IDE or from the command line.
    """
    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    consoleHandler = logging.StreamHandler()
    logger.addHandler(consoleHandler)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    consoleHandler.setFormatter(formatter)
    logging.info('Logger enabled and logging')
