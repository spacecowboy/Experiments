'''
Created on Sep 1, 2011

@author: jonask
'''

from kalderstam.util.filehandling import parse_file, get_cross_validation_sets
from survival.network import build_feedforward_committee
import numpy
from survival.cox_error_in_c import get_C_index
from survival.cox_genetic import c_index_error
import logging
from kalderstam.neural.training.davis_genetic import train_evolutionary
#from kalderstam.neural.training.genetic import train_evolutionary

try:
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab
    from survival.plotting import kaplanmeier
except ImportError:
    plt = None
except RuntimeError:
    plt = None

from kalderstam.neural.training.committee import train_committee
import sys

def com_cross():

    filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset.txt"

    #try:
    #    columns = input("Which columns to include? (Do NOT forget trailing comma if only one column is used, e.g. '3,'\nAvailable columns are: 2, -4, -3, -2, -1. Just press ENTER for all columns.\n")
    #except SyntaxError:
    #if len(sys.argv) < 3:
    columns = (2, -4, -3, -2, -1)
    #else:
    #    columns = [int(col) for col in sys.argv[2:]]

    print('\nIncluding columns: ' + str(columns))

    P, T = parse_file(filename, targetcols = [4, 5], inputcols = columns, ignorerows = [0], normalize = True)
    #remove tail censored
    #print('\nRemoving tail censored...')
    #P, T = copy_without_censored(P, T)

    #Divide into validation sets
    #test_size = 0.33
    #print('Size of test set (not used in training): ' + str(test_size))
    #((TP, TT), (VP, VT)) = get_validation_set(P, T, validation_size = test_size, binary_column = 1)

    print("\nData set:")
    print("Number of patients with events: " + str(T[:, 1].sum()))
    print("Number of censored patients: " + str((1 - T[:, 1]).sum()))

    #print("Length of training set: " + str(len(TP)))
    #print("Length of test set: " + str(len(VP)))

    #try:
    #    comsize = input("Number of networks to cross-validate [10]: ")
    #except SyntaxError:
    if len(sys.argv) < 2:
        netsize = 1
    else:
        netsize = int(sys.argv[1])
    print("\nNumber of hidden nodes: " + str(netsize))
    comsize = 4
    print('Number of members in each committee: ' + str(comsize))
    comnum = 5
    print('Number of committees to cross-validate: ' + str(comnum))

    times_to_cross = 3
    print('Number of times to repeat cross-validation: ' + str(times_to_cross))

    #try:
    #    pop_size = input('Population size [50]: ')
    #except SyntaxError as e:
    pop_size = 100
    print("Population size: " + str(pop_size))

    #try:
    #    mutation_rate = input('Please input a mutation rate (0.25): ')
    #except SyntaxError as e:
    mutation_rate = 0.05
    print("Mutation rate: " + str(mutation_rate))

    #try:
    #    epochs = input("Number of generations (200): ")
    #except SyntaxError as e:
    epochs = 100
    print("Epochs: " + str(epochs))

    for _cross_time in xrange(times_to_cross):

        data_sets = get_cross_validation_sets(P, T, comnum , binary_column = 1)

        print('\nTest Errors, Validation Errors:')

        for _com_num, (TS, VS) in zip(xrange(comnum), data_sets):
            com = build_feedforward_committee(comsize, len(P[0]), netsize, 1, output_function = 'linear')

            #1 is the column in the target array which holds the binary censoring information
            test_errors, vald_errors, internal_sets = train_committee(com, train_evolutionary, TS[0], TS[1], 1, epochs, error_function = c_index_error, population_size = pop_size, mutation_chance = mutation_rate)

            com.set_training_sets([set[0][0] for set in internal_sets]) #first 0 gives training sets, second 0 gives inputs.

            outputs = numpy.array([[com.risk_eval(inputs)] for inputs in TS[0]]) #Need double brackets for dimensions to be right for numpy
            train_c_index = get_C_index(TS[1], outputs)
            outputs = numpy.array([[com.risk_eval(inputs)] for inputs in VS[0]]) #Need double brackets for dimensions to be right for numpy
            val_c_index = get_C_index(VS[1], outputs)

            print(str(1.0 / train_c_index) + ", " + str(1.0 / val_c_index))

if __name__ == '__main__':
    com_cross()
