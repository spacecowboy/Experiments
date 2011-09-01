'''
Created on Sep 1, 2011

@author: jonask
'''
from kalderstam.util.filehandling import parse_file, get_validation_set, \
    read_data_file
from survival.network import build_feedforward_committee
import numpy
from survival.cox_error_in_c import get_C_index
from survival.cox_genetic import c_index_error
import logging
from kalderstam.neural.training.davis_genetic import train_evolutionary
import os
import random
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
logger = logging.getLogger('kalderstam.neural.cox_training')

def plot_input(data):
    if plt:
        plt.figure()
        plt.title("Mean: " + str(numpy.mean(data)) + " Std: " + str(numpy.std(data)))
        n, bins, patches = plt.hist(data, 50, normed = 1, facecolor = 'green', alpha = 0.75)
        # add a 'best fit' line
        y = mlab.normpdf(bins, numpy.mean(data), numpy.std(data))
        l = plt.plot(bins, y, 'r--', linewidth = 1)

def copy_without_censored(Porg, Torg, cutoff = 0):
    '''
    Cutoff = 5 (default), specifies up to which time censored data is allowed to in the data set.
    Note that this limit is non-inclusive. If cutoff = 5, then all censored data with time >= 5 will be removed.
    '''
    P = Porg.copy()
    T = Torg.copy()
    indices = []
    for index in xrange(len(T)):
        if T[index, 1] or T[index, 0] < cutoff:
            indices.append(index)

    T = T[indices]
    P = P[indices]

    return P, T

def committee_test():

    try:
        netsize = input('Number of hidden nodes? [1]: ')
    except SyntaxError as e:
        netsize = 1

    try:
        comsize = input('Committee size? [1]: ')
    except SyntaxError as e:
        comsize = 1

    try:
        pop_size = input('Population size? [100]: ')
    except SyntaxError as e:
        pop_size = 100

    try:
        mutation_rate = input('Please input a mutation rate (0.05): ')
    except SyntaxError as e:
        mutation_rate = 0.05

    filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset.txt"

    try:
        columns = input("Which columns to include? (Do NOT forget trailing comma if only one column is used, e.g. '3,'\nAvailable columns are: 2, -4, -3, -2, -1. Just press ENTER for all columns.\n")
    except SyntaxError:
        columns = (2, -4, -3, -2, -1)

    P, T = parse_file(filename, targetcols = [4, 5], inputcols = columns, ignorerows = [0], normalize = True)

    #remove tail censored
    try:
        cutoff = input('Cutoff for censored data? [9999 years]: ')
    except SyntaxError as e:
        cutoff = 9999
    P, T = copy_without_censored(P, T, cutoff)

    #Divide into validation sets
    try:
        test_size = float(input('Size of test set (not used in training)? Input in fractions. Default is [0.0]: '))
    except:
        test_size = 0.0
    ((TP, TT), (VP, VT)) = get_validation_set(P, T, validation_size = test_size, binary_column = 1)
    print("Length of training set: " + str(len(TP)))
    print("Length of test set: " + str(len(VP)))

    try:
        epochs = input("\nNumber of generations (1): ")
    except SyntaxError as e:
        epochs = 1

    com = build_feedforward_committee(comsize, len(P[0]), netsize, 1, output_function = 'linear')

    #1 is the column in the target array which holds the binary censoring information
    test_errors, vald_errors, data_sets = train_committee(com, train_evolutionary, P, T, 1, epochs, error_function = c_index_error, population_size = pop_size, mutation_chance = mutation_rate)

    com.set_training_sets([set[0][0] for set in data_sets]) #first 0 gives training sets, second 0 gives inputs.

    print('\nTest C_indices, Validation C_indices:')
    for terr, verr in zip(test_errors.values(), vald_errors.values()):
        print(str(1 / terr) + ", " + str(1 / verr))

    if plt:
        outputs = numpy.array([[com.risk_eval(inputs)] for inputs in TP]) #Need double brackets for dimensions to be right for numpy
        kaplanmeier(time_array = TT[:, 0], event_array = TT[:, 1], output_array = outputs[:, 0], threshold = 0.5)
        train_c_index = get_C_index(TT, outputs)
        print("\nC-index on the training set: " + str(train_c_index))
        if len(VP) > 0:
            outputs = numpy.array([[com.risk_eval(inputs)] for inputs in VP]) #Need double brackets for dimensions to be right for numpy
            test_c_index = get_C_index(VT, outputs)
            kaplanmeier(time_array = VT[:, 0], event_array = VT[:, 1], output_array = outputs[:, 0], threshold = 0.5)
            print("C-index on the test set: " + str(test_c_index))

        #raw_input("\nPress enter to show plots...")
        plt.show()

    try:
        answer = input("\nDo you wish to print committee risk output? ['n']: ")
    except (SyntaxError, NameError):
        answer = 'n'

    if answer != 'n' and answer != 'no':
        inputs = read_data_file(filename)
        P, T = parse_file(filename, targetcols = [4, 5], inputcols = columns, ignorerows = [0], normalize = True)
        outputs = [[com.risk_eval(patient)] for patient in P]
        while len(inputs) > len(outputs):
            outputs.insert(0, ["net_output"])

        print("\n")
        for rawline in zip(inputs, outputs):
            line = ''
            for col in rawline[0]:
                line += str(col)
                line += ','
            for col in rawline[1]:
                line += str(col)

            print(line)

if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG)

    committee_test()
