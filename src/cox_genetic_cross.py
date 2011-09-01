from kalderstam.util.filehandling import parse_file
from kalderstam.neural.network import build_feedforward_committee
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

def cross_validation_test():

    filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset.txt"

    #try:
    #    columns = input("Which columns to include? (Do NOT forget trailing comma if only one column is used, e.g. '3,'\nAvailable columns are: 2, -4, -3, -2, -1. Just press ENTER for all columns.\n")
    #except SyntaxError:
    if len(sys.argv) < 3:
        columns = (2, -4, -3, -2, -1)
    else:
        columns = [int(col) for col in sys.argv[2:]]

    print('\nIncluding columns: ' + str(columns))

    P, T = parse_file(filename, targetcols = [4, 5], inputcols = columns, ignorerows = [0], normalize = True)
    #remove tail censored
    #print('\nRemoving tail censored...')
    #P, T = copy_without_censored(P, T)

    print("\nData set:")
    print("Number of patients with events: " + str(T[:, 1].sum()))
    print("Number of censored patients: " + str((1 - T[:, 1]).sum()))

    #try:
    #    comsize = input("Number of networks to cross-validate [10]: ")
    #except SyntaxError:
    comsize = 5
    print('\nNumber of networks to cross-validate: ' + str(comsize))

    times_to_cross = 3
    print('\nNumber of times to repeat cross-validation: ' + str(times_to_cross))

    #try:
    #    netsize = input('Number of hidden nodes [3]: ')
    #except SyntaxError as e:
    if len(sys.argv) < 2:
        netsize = 1
    else:
        netsize = int(sys.argv[1])
    print("Number of hidden nodes: " + str(netsize))

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
    epochs = 400
    print("Epochs: " + str(epochs))

    for _ in xrange(times_to_cross):
        com = build_feedforward_committee(comsize, len(P[0]), netsize, 1, output_function = 'linear')

        #1 is the column in the target array which holds the binary censoring information
        test_errors, vald_errors, data_sets = train_committee(com, train_evolutionary, P, T, 1, epochs, error_function = c_index_error, population_size = pop_size, mutation_chance = mutation_rate)

        print('\nTest Errors, Validation Errors:')
        for terr, verr in zip(test_errors.values(), vald_errors.values()):
            print(str(terr) + ", " + str(verr))


if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG)

    cross_validation_test()
