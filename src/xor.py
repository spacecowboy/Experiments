'''
Created on Jun 7, 2011

@author: jonask
'''
from kalderstam.neural.error_functions.sum_squares import total_error
from kalderstam.neural.network import build_feedforward
from kalderstam.util.filehandling import parse_data
from kalderstam.neural.training.gradientdescent import traingd
import numpy

xor_set = [[0, 0, 0],
           [0, 1, 1],
           [1, 0, 1],
           [1, 1, 0]]

xor_set = numpy.array(xor_set)

P, T = parse_data(xor_set, targetcols = 2, inputcols = [0, 1], normalize = False)

net = build_feedforward(2, 4, 1)

print("Error before training: " + str(total_error(T, net.sim(P))))
net = traingd(net, (P, T), (None, None), epochs = 1000, learning_rate = 0.1, block_size = 0)
print("Error after training: " + str(total_error(T, net.sim(P))))
