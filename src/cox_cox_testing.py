from kalderstam.matlab.matlab_functions import plot_network_weights
from kalderstam.util.filehandling import parse_file, load_network, save_network, \
    get_cross_validation_sets
from kalderstam.neural.network import build_feedforward
import time
import numpy
import matplotlib.pyplot as plt
from survival.cox_error import calc_sigma, calc_beta, generate_timeslots, \
    censor_rndtest, orderscatter
from survival.cox_error_in_c import get_C_index
import survival.cox_error as cox_error
import kalderstam.util.graphlogger as glogger
import logging
from kalderstam.neural.training.gradientdescent import traingd
from survival.plotting import kaplanmeier
import sys

logger = logging.getLogger('kalderstam.neural.cox_training')

def experiment(net, P, T, vP, vT, filename, epochs, learning_rate):
    logger.info("Running experiment for: " + filename + ' ' + str(epochs) + ", rate: " + str(learning_rate))
    print("Number of patients with events: " + str(T[:, 1].sum()))
    print("Number of censored patients: " + str((1 - T[:, 1]).sum()))

    timeslots = generate_timeslots(T)

    try:
        net = traingd(net, (P, T), (vP, vT), epochs, learning_rate, block_size = 100, error_module = cox_error)
    except FloatingPointError:
        print('Aaawww....')
    outputs = net.sim(P)
    c_index = get_C_index(T, outputs)
    logger.info("C index = " + str(c_index))

    #plot_network_weights(net)

    kaplanmeier(time_array = T[:, 0], event_array = T[:, 1], output_array = outputs[:, 0])
    if vP is not None and len(vP) > 0:
        outputs = net.sim(vP)
        kaplanmeier(time_array = vT[:, 0], event_array = vT[:, 1], output_array = outputs[:, 0])

    return net

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    glogger.setLoggingLevel(glogger.nothing)

    filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset.txt"

    #try:
    #    columns = input("Which columns to include? (Do NOT forget trailing comma if only one column is used, e.g. '3,'\nAvailable columns are: 2, -4, -3, -2, -1. Just press ENTER for all columns.\n")
    #except SyntaxError:
    columns = (2, -4, -3, -2, -1)
    print('\nIncluding columns: ' + str(columns))

    P, T = parse_file(filename, targetcols = [4, 5], inputcols = columns, ignorerows = [0], normalize = True)
    #remove tail censored
    #print('\nRemoving tail censored...')
    #P, T = copy_without_tailcensored(P, T)

    try:
        pieces = input('Number of crossvalidation pieces? [1]: ')
    except SyntaxError as e:
        pieces = 1

    #Divide into validation sets
    TandV = get_cross_validation_sets(P, T, pieces , binary_column = 1)

    for set, ((tP, tT), (vP, vT)) in zip(range(pieces), TandV):
        print("\nCross validation set " + str(set))
        print("Training")
        print("Number of patients with events: " + str(tT[:, 1].sum()))
        print("Number of censored patients: " + str((1 - tT[:, 1]).sum()))
        print("Validation")
        print("Number of patients with events: " + str(vT[:, 1].sum()))
        print("Number of censored patients: " + str((1 - vT[:, 1]).sum()))

    try:
        netsize = input('\nNumber of hidden nodes? [3]: ')
    except SyntaxError as e:
        netsize = 3

    try:
        blocksize = input('Blocksize? [100]: ')
    except SyntaxError as e:
        blocksize = 100

    try:
        epochs = input("Number of epochs (200): ")
    except SyntaxError as e:
        epochs = 200

    try:
        rate = input('Learning rate? [1]: ')
    except SyntaxError as e:
        rate = 1

    #Network part
    p = len(P[0]) #number of input covariates

    for set, ((tP, tT), (vP, vT)) in zip(range(pieces), TandV):

        net = build_feedforward(p, netsize, 1, output_function = 'linear')

        net = experiment(net, tP, tT, vP, vT, filename, epochs, rate)

    plt.show()
