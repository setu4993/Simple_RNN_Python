from .file_io import *
from .network import rnn_base, rnn_train, rnn_predict

"""
Descriptions for most recurring variables across functions.

Input parameters:
-----------------

- inputs:       List of `[day, temp, prec]` lists containing the real-world values of the input variables. Ranges 
                (except day) do not correspond to maximum or minimum possible, but approximately to 
                [- 2.5 standard deviations, + 2.5 standard deviation].
    -- day:     Integer value for day of the year. Range: [0-365]
    -- temp:    Integer value of maximum temperature for the day, as tenths of degrees Celsius. For 30.0C, the value of 
                this variable should be 300. Range: [-100, 300]
    -- prec:    Integer value for precipitation (sum of rainfall and snowfall) of the day, in mm. Range: [0, 252]
- networks:     List containing the index of networks these inputs should run on. Typically corresponds to the month of 
                the year the prediction is for. The length of this list is the same as the length of input list. 
                Range: [0, 11]
- targets:      List containing target values for the given inputs. The length of this list is the same as the length of 
                input list.

Output parameters:
------------------

- predictions:  List containing predicted values for the given inputs. The length of this list is the same as the length 
                of input list.

Network parameters:
-------------------

- inp:          List of real-world `[day, temp, prec]` values.
- wih:          Weights for the neural network connecting the input-hidden layer. 3D array of type `np.ndarray` with 
                dimensions `[i, j, k]` where `i` is the number of networks, `j` is the number of input nodes, `k` is the number of hidden 
                layer nodes.
- whh:          Weights for the neural network connecting the previous hidden values-hidden layer. 3D array of type 
                `np.ndarray` with dimensions `[i, j, k]` where `i` is the number of networks, `j` is the number of 
                previous hidden layer nodes, `k` is the number of hidden layer nodes.
- who:          Weights for the neural network connecting the hidden-output layer. 3D array of type `np.ndarray` with 
                dimensions `[i, j, k]` where `i` is the number of networks, `j` is the number of hidden layer nodes, `k` 
                is the number of output layer nodes (expected to be 1).
- prh:          Previous hidden layer values. 2D array of type `np.ndarray` of dimensions `[i, j]` where `i` is the 
                number of networks, `j` is the number of previous hidden layer nodes.        
- input_nodes:  A `np.ndarray` of pre-processed input values that are used as an input to the network.
- network:      Integer value specifying the network to use for making the prediction between [0, 11].
- alpha:        Alpha value for back-propagation, typically in the range of `(0, 1)`. Use lower values for training and 
                higher values during online testing.


Miscellaneous parameters:
-------------------------
- dump_location: Location of the weights pickle file on disk.
- log:           Boolean flag indicating if logging to console should be enabled.
"""

__authors__ = 'Setu Shah'
__version = '0.4'
