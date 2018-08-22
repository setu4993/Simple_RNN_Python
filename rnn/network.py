import numpy as np
from .file_io import *
import logging


class rnn_base:
    """
    This class implements the recurrent neural network.
    """
    def __init__(self, wih, whh, who, prh, alpha=0.38, h_bias=True, log=False):
        """
        This function initializes the various parameters of the recurrent neural network.

        :param wih, whh, who, prh, alpha
        :param h_bias:              [Optional] Boolean variable indicating if hidden layer consists of a bias node
                                    (pre-trained model includes a bias node). Defaults to True.
        :param log:                 [Optional] Boolean variable indicating if logging is to be enabled.
        """

        self.TOTAL_NETWORKS = wih.shape[0]

        self.INPUT_NODES_NUM = wih.shape[1] - 1
        self.HIDDEN_NODES_NUM = wih.shape[2]
        self.OUTPUT_NODES_NUM = who.shape[2]

        self.wih = wih
        self.whh = whh
        self.who = who

        self.prev_nodes = prh

        self.hidden_nodes = np.zeros(self.HIDDEN_NODES_NUM)
        self.output_nodes = np.zeros(self.OUTPUT_NODES_NUM)


        self.ALPHA = alpha

        self.h_bias = h_bias

        self.delta_ho = np.empty(0)
        self.delta_who = np.empty(0)
        self.delta_in = np.empty(0)
        self.delta_ih = np.empty(0)

        if log:
            log_to_console()

        logging.info('RNN initialized')

    def load(self, weights_file_location):
        """
        This function loads the weights file by using the functions from `file_io.py`, based on input provided.

        :param weights_file_location: Same as `weights` from the `__init__` function.
        :return: [wih, whh, who, prh]
        """
        if weights_file_location.endswith('.csv'):
            return load_weights_from_csv(weights_file_location)
        else:
            return load_weights_from_pickle_dump(weights_file_location)

    def save(self, dump_location):
        """
        This function saves the weights to the pickle file location specified.
        :param dump_location
        """
        save_weights_to_pickle_dump(dump_location, self.wih, self.whh, self.who, self.prev_nodes)
        logging.info('Weights dumped to pickle file')

    def calc_hidden(self, input_nodes, network):
        """
        Calculates the hidden layer of the network.

        :params input_nodes, network:
        """
        self.hidden_nodes += self.calc_next_layer(input_nodes, self.wih[network])
        self.hidden_nodes += self.calc_next_layer(self.prev_nodes[network], self.whh[network])
        if self.h_bias:
            self.hidden_nodes = np.concatenate((self.hidden_nodes, np.ones(1)))

    def calc_output(self, network):
        """
        Calculates the output layer of the network.
        :param network
        """
        self.output_nodes += self.calc_next_layer(self.hidden_nodes, self.who[network])

    def calc_next_layer(self, current_nodes, weights):
        """
        Calculates the next layer of the network.
        :param current_nodes: A 1D `np.ndarray` containing values of the nodes of the current layer.
        :param weights: A 2D `np.ndarray` containing weights connecting the current layer to the next layer.
        :return: A 1D `np.ndarray` containing the values of the next layer of nodes.
        """
        return np.tanh(np.sum((weights.T * current_nodes).T, axis=0))

    def clear(self):
        """
        Clears the arrays whose values change after every iteration.
        """
        self.hidden_nodes = np.zeros(self.HIDDEN_NODES_NUM)
        self.output_nodes = np.zeros(self.OUTPUT_NODES_NUM)
        self.delta_ho = np.empty(0)
        self.delta_who = np.empty(0)
        self.delta_in = np.empty(0)
        self.delta_ih = np.empty(0)

    def upscale_output(self, value):
        """
        Upscales the output values to show the real predictions from the normalized prediction.

        :param value: `np.ndarray` containing output from the network.
        :return: prediction
        """
        return np.arctanh(value) * 20 + 140

    def downscale_output(self, value):
        """
        Downscales the target value to a normalized value.

        :param value: Float target value.
        :return: target: `np.ndarray` containing normalized target value.
        """
        return np.tanh((value - 140) / 20)

    def pre_process_input_values(self, inp):
        """
        Pre-processes and scales the input values to use as input to the RNN. Without this step, the input will produce
        incorrect results.

        :param inp
        :return: input_nodes
        """
        if inp[0] > 365:
            raise ValueError('Day of the year cannot be greater than 365')
        elif inp[0] > 183:
            inp[0] = 366 - inp[0]
        nodes = np.zeros(self.INPUT_NODES_NUM)
        nodes[0] = np.tanh((inp[0] - 92) / 46)
        nodes[1] = np.tanh((inp[1] - 100) / 100)
        nodes[2] = np.tanh((inp[2] - 126) / 63)
        return np.concatenate((nodes, np.ones(1)))

    def backprop(self, input_nodes, target, network):
        """
        This function implements the back-propagation that allows the network to learn from the errors in predictions.

        :param input_nodes, target:, network
        """
        self.delta_ho = (target - self.output_nodes) * (1 - np.square(np.tanh(self.output_nodes)))
        self.delta_who = (self.ALPHA * self.delta_ho * self.hidden_nodes).reshape(self.HIDDEN_NODES_NUM + 1, 1)
        self.delta_in = self.delta_ho * self.who[network]
        self.delta_ih = self.delta_in * (1 - np.square(np.tanh(self.hidden_nodes))).reshape(self.HIDDEN_NODES_NUM + 1, 1)

        self.wih[network] += (self.ALPHA * self.delta_ih[:self.HIDDEN_NODES_NUM] * input_nodes).T
        self.whh[network] += (self.ALPHA * self.delta_ih[:self.HIDDEN_NODES_NUM] * self.prev_nodes[network]).T
        self.who[network] += self.delta_who
        self.prev_nodes[network] = self.hidden_nodes[:self.HIDDEN_NODES_NUM]

    def network_state(self):
        """
        This function returns the network state that can be used as parameters for creating a prediction network.

        :return: A list containing [wih, whh, who, prh].
        """
        return [self.wih, self.whh, self.who, self.prev_nodes]


class rnn_train(rnn_base):
    """
    This is a sub-class of the `rnn_base` class, which implements the training methods.
    """
    def __init__(self, input_nodes_num, hidden_nodes_num, output_nodes_num, total_networks=1, alpha=0.25, i_bias=True,
                 h_bias=True, log=False, random_seed=12314):
        """
        This function initializes the parameters for this class and the parent class.

        :param input_nodes_num:     Integer specifying number of input nodes.
        :param hidden_nodes_num:    Integer specifying number of hidden nodes.
        :param output_nodes_num:    Integer specifying number of output nodes.
        :param total_networks:      [Optional] Number of networks to train over. Defaults to 1.
        :param i_bias:              [Optional] Boolean variable indicating if input layer consists of a bias node.
                                    Defaults to True.
        :param alpha, h_bias, log
        :param random_seed:         [Optional] Integer value for specifying the random seed (use the same seed for
                                    reproducibility). Defaults to 12314.
        """
        if i_bias:
            input_nodes_num += 1
        if h_bias:
            hidden_nodes_num += 1

        np.random.seed(random_seed)

        wih = 2 * np.random.random((total_networks, input_nodes_num, hidden_nodes_num - 1)) - 1
        whh = 2 * np.random.random((total_networks, hidden_nodes_num - 1, hidden_nodes_num - 1)) - 1
        who = 2 * np.random.random((total_networks, hidden_nodes_num, output_nodes_num)) - 1

        prh = np.zeros((total_networks, hidden_nodes_num - 1))

        logging.info('Random arrays created')

        rnn_base.__init__(self, wih, whh, who, prh, alpha=alpha, h_bias=h_bias, log=log)

    def train_single(self, inp, network, target):
        """
        This function implements training for one set of input, network and target values.

        :param inp, network, target
        """
        input_nodes = self.pre_process_input_values(inp)
        self.calc_hidden(input_nodes, network)
        self.calc_output(network)
        self.backprop(input_nodes, self.downscale_output(target), network)

    def train_many(self, inputs, networks, targets):
        """
        This function makes predictions for lists of inputs, networks and targets.

        :params inputs, networks, targets
        """
        for i, (inp, network, target) in enumerate(zip(inputs, networks, targets)):
            self.train_single(inp, network, target)
            self.clear()

    def train(self, inputs, targets, networks=None, epochs=50000):
        """
        This function implements the training for a given number of epochs and is the preferred interface to train the
        network.

        :param inputs, targets, networks
        :param epochs:  [Optional] Number of epochs to train the network for. Defaults to 50,000.
        """
        if not networks:
            networks = np.zeros(len(inputs), np.int32)
        else:
            assert len(inputs) == len(networks), "Length of inputs and networks not equal"

        if isinstance(inputs[-1], int) or len(inputs) == 1:
            logging.warning('Network is trained best when multiple inputs are specified')

        assert len(inputs) == len(targets), "Length of inputs and targets not equal"

        for i in range(epochs):
            self.train_many(inputs, networks, targets)
            if i % 1000 == 0:
                logging.info('Epoch %d completed' % i)
        logging.info('RNN trained')


class rnn_predict(rnn_base):
    """
    This is a sub-class of the `rnn_base` class, which implements the prediction methods.
    """
    def __init__(self, weights, alpha=0.38, h_bias=True, log=False):
        """
        This function initializes the parameters for this class and the parent class.

        :param weights: Either a string that points to the location of the weights pickle file, or the list from the
                        un-pickled weights file. List should be of the format `[wih, whh, who, prh]`.
        :param alpha, h_bias, log
        """
        if isinstance(weights, str):
            [wih, whh, who, prh] = self.load(weights)
            logging.info('Weights loaded from pickle file')
        elif isinstance(weights, list):
            [wih, whh, who, prh] = weights
            logging.info('Weights loaded from passed array')
        else:
            raise TypeError('Incorrect type passed to the class')

        rnn_base.__init__(self, wih, whh, who, prh, alpha=alpha, h_bias=h_bias, log=log)

    def predict_single(self, inp, network, recal=False, target=None):
        """
        Makes a prediction for one set of input values.

        :param inp, network
        :return: prediction
        """
        input_nodes = self.pre_process_input_values(inp)
        self.calc_hidden(input_nodes, network)
        self.calc_output(network)
        if recal and target:
            self.backprop(input_nodes, self.downscale_output(target), network)
        return self.upscale_output(self.output_nodes)[0]

    def predict_many(self, inputs, networks, recal=False, targets=None):
        """
        This function makes predictions if a list of lists in given as input.

        :params inputs, networks, targets, recal
        :return: predictions
        """
        logging.info('Making multiple predictions')
        predictions = []
        for i, inp in enumerate(inputs):
            if not recal:
                predictions.append(self.predict_single(inp, networks[i], recal=recal))
                print('The prediction for inputs ', inp, ' for network ', networks[i], ' is %.2f' % predictions[-1])
            else:
                predictions.append(self.predict_single(inp, networks[i], recal=recal, target=targets[i]))
                print('The target for inputs ', inp, ' for network ', networks[i], ' was ', targets[i],
                      'and the prediction was %.2f, error observed was %.2f%%. ' % (predictions[-1],
                                                                                 abs(predictions[-1] - targets[i]) * 100
                                                                                 / targets[i]))
            self.clear()
        logging.info('All predictions made')
        return predictions

    def predict(self, inputs, networks=None, targets=None, recal=False):
        """
        This function is the primary interface of the class. This is the function that should be called to make
        predictions and for back-propagation.

        :param inputs, networks, targets
        :param recal: Boolean value specifying if the network should be re-calibrated. Requires the `targets` list to be
                      specified.
        :return: predictions
        """
        if not networks:
            networks = np.zeros(len(inputs), np.int32)
        if recal and not targets:
                raise AttributeError('Attribute `targets` not found')
        if isinstance(inputs[-1], list):
            predictions = self.predict_many(inputs, networks, recal=recal, targets=targets)
        elif isinstance(inputs[-1], int):
            predictions = [self.predict_single(inputs, networks, recal=recal, targets=targets)]
        else:
            raise TypeError('Type of `inputs` incorrect')
        return predictions
