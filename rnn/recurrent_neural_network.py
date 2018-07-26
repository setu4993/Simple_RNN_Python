from file_io import *


class recurrent_neural_network:
    """
    This class implements the recurrent neural network and all of the functions for the network.
    """
    def __init__(self, weights, alpha=0.38, h_bias=True):
        """
        This function initializes the various parameters of the recurrent neural network.

        :param weights: Either a string that points to the location of the weights pickle file, or the list from the
                        un-pickled weights file. List should be of the format `[wih, whh, who, prh]`.
        :param alpha:   [Optional] Alpha value for back-propagation, typically in the range of `(0, 1)`.
                        Defaults to 0.38.
        :param h_bias:  [Optional] boolean variable indicating if hidden layer consists of a bias node (pre-trained
                        model includes a bias node). Defaults to True.
        """
        if isinstance(weights, str):
            [wih, whh, who, prh] = self.load(weights)
        elif isinstance(weights, list):
            [wih, whh, who, prh] = weights
        else:
            raise TypeError('Incorrect type passed to the class')
        self.TOTAL_NETWORKS = wih.shape[0]

        self.INPUT_NODES_NUM = wih.shape[1] - 1
        self.HIDDEN_NODES_NUM = wih.shape[2]
        self.OUTPUT_NODES_NUM = who.shape[2]

        self.hidden_nodes = np.zeros(self.HIDDEN_NODES_NUM)
        self.output_nodes = np.zeros(self.OUTPUT_NODES_NUM)

        self.wih = wih
        self.whh = whh
        self.who = who

        self.prev_nodes = prh

        self.ALPHA = alpha

        self.h_bias = h_bias

        self.delta_ho = np.empty(0)
        self.delta_who = np.empty(0)
        self.delta_in = np.empty(0)
        self.delta_ih = np.empty(0)

    def load(self, file_location):
        if file_location.endswith('.csv'):
            return load_weights_from_csv(file_location)
        else:
            return load_weights_from_pickle_dump(file_location)

    def save(self, file_location):
        save_weights_to_pickle_dump(file_location, self.wih, self.whh, self.who, self.prh)

    def calc_hidden(self, input_nodes, network):
        self.hidden_nodes += self.calc_next_layer(input_nodes, self.wih[network])
        self.hidden_nodes += self.calc_next_layer(self.prev_nodes[network], self.whh[network])
        if self.h_bias:
            self.hidden_nodes = np.concatenate((self.hidden_nodes, np.ones(1)))

    def calc_output(self, network):
        self.output_nodes += self.calc_next_layer(self.hidden_nodes, self.who[network])

    def calc_next_layer(self, current_nodes, weights):
        return np.tanh(np.sum((weights.T * current_nodes).T, axis=0))

    def clear(self):
        self.hidden_nodes = np.zeros(self.HIDDEN_NODES_NUM)
        self.output_nodes = np.zeros(self.OUTPUT_NODES_NUM)
        self.delta_ho = np.empty(0)
        self.delta_who = np.empty(0)
        self.delta_in = np.empty(0)
        self.delta_ih = np.empty(0)

    def upscale(self, value):
        return np.arctanh(value) * 20 + 140

    def downscale(self, value):
        return np.tanh((value - 140) / 20)

    def pre_process_input_values(self, inp):
        """
        Pre-processes and scales the input values to use as input to the RNN. Without this step, the input will produce
        incorrect results.

        :param day: Day of the year: Integer in the range of 0-365.
        :param temp: Maximum temperature: Integer value of maximum temperature for the day, as tenths of degrees Celsius.
        For 30.0C, the value of this variable should be 300.
        :param prec: Precipitation (sum of rainfall and snowfall): Integer value for precipitation of the day, in mm.
        :return: A `np.ndarray` of scaled input values that can be used as an input to the network.
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

    def predict_single(self, inp, network, recal=False, target=None):
        """
        Makes a prediction for one set of input values.

        :param inp: A list of input values of length `INPUT_NODES_NUM + 1`.
        :param network: The network to use for making the prediction. [Ideally, between `0` and `11`, each corresponding
                        to one month from January (0) - December (11).]
        :return: The output prediction.
        """
        input_nodes = self.pre_process_input_values(inp)
        self.calc_hidden(input_nodes, network)
        self.calc_output(network)
        if recal and target:
            self.backprop(input_nodes, self.downscale(target), network)
        return self.upscale(self.output_nodes)[0]

    def predict_many(self, inputs, networks, recal=False, targets=None):
        predictions = []
        for i, inp in enumerate(inputs):
            if not recal:
                predictions.append(self.predict_single(inp, networks[i], recal))
                print('The output for inputs ', inp, ' is: %.2f' % predictions[-1])
            else:
                self.clear()
                predictions.append(self.predict_single(inp, networks[i], recal, targets[i]))
                print('The target for inputs ', inp, ' for network ', networks[i], ' was ', targets[i],
                      'and the prediction was: %.2f' % predictions[-1])
        return predictions

    def backprop(self, input_nodes, target, network):
        self.delta_ho = (target - self.output_nodes) * (1 - np.square(np.tanh(self.output_nodes)))
        self.delta_who = (self.ALPHA * self.delta_ho * self.hidden_nodes).reshape(self.HIDDEN_NODES_NUM + 1, 1)
        self.delta_in = self.delta_ho * self.who[network]
        self.delta_ih = self.delta_in * (1 - np.square(np.tanh(self.hidden_nodes))).reshape(self.HIDDEN_NODES_NUM + 1, 1)

        self.wih[network] += (self.ALPHA * self.delta_ih[:self.HIDDEN_NODES_NUM] * input_nodes).T
        self.whh[network] += (self.ALPHA * self.delta_ih[:self.HIDDEN_NODES_NUM] * self.prev_nodes[network]).T
        self.who[network] += self.delta_who
        self.prev_nodes[network] = self.hidden_nodes[:self.HIDDEN_NODES_NUM]

    def run_network(self, inputs, networks=None, targets=None, recal=False):
        if not networks:
            networks = np.zeros(len(inputs), np.int32)
        if recal and not targets:
                raise AttributeError('Attribute `targets` not found')
        if isinstance(inputs[-1], list):
            predictions = self.predict_many(inputs, networks, recal, targets)
        elif isinstance(inputs[-1], int):
            predictions = self.predict_single(inputs, networks, recal, targets)
        else:
            raise TypeError('Type of `inputs` incorrect')
        return predictions
