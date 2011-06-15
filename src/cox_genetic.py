from kalderstam.matlab.matlab_functions import plot_network_weights
from kalderstam.util.filehandling import parse_file, load_network, save_network
from kalderstam.neural.network import build_feedforward
import numpy
from survival.cox_error import orderscatter, get_C_index
import survival.cox_error as cox_error
from survival.cox_error import censor_rndtest
import kalderstam.util.graphlogger as glogger
import logging
from kalderstam.neural.training.gradientdescent import traingd
from kalderstam.neural.training.genetic import train_evolutionary
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import sample
logger = logging.getLogger('kalderstam.neural.cox_training')

def c_index_error(target, result):
    #len(target) first to compensate for internals in genetic training
    #abs( - 0.5) to make both "positive" and "negative" C_index work, since they do
    C = get_C_index(target, result)
    if C < 0.5:
        #dont want these right now
        C = 0.49 #this is bad

    return len(target) / abs(C - 0.5) - 2 * len(target) #return inverse, error should be low if c_index is high. last minus term makes the minimum zero and not two.

def plot_input(data):
    plt.figure()
    plt.title("Mean: " + str(numpy.mean(data)) + " Std: " + str(numpy.std(data)))
    n, bins, patches = plt.hist(data, 50, normed = 1, facecolor = 'green', alpha = 0.75)
    # add a 'best fit' line
    y = mlab.normpdf(bins, numpy.mean(data), numpy.std(data))
    l = plt.plot(bins, y, 'r--', linewidth = 1)

def copy_without_tailcensored(Porg, Torg, cutoff = 5):
    P = Porg.copy()
    T = Torg.copy()
    indices = []
    for index in xrange(len(T)):
        if T[index, 1] or T[index, 0] < cutoff:
            indices.append(index)

    T = T[indices]
    P = P[indices]

    return P, T

def test(net, P, T, filename, epochs):
    logger.info("Running genetic test for: " + filename + ' ' + str(epochs))
    print("Number of patients with events: " + str(T[:, 1].sum()))
    print("Number of censored patients: " + str((1 - T[:, 1]).sum()))

    outputs = net.sim(P)
    c_index = get_C_index(T, outputs)
    logger.info("C index = " + str(c_index))

    try:
        #net = train_cox(net, (P, T), (None, None), timeslots, epochs, learning_rate = learning_rate)
        #net = traingd(net, (P, T), (None, None), epochs, learning_rate, block_size, error_module = cox_error)
        net = train_evolutionary(net, (P, T), (None, None), epochs / 2, error_function = c_index_error, population_size = 50)

        outputs = net.sim(P)
        c_index = get_C_index(T, outputs)
        logger.info("C index = " + str(c_index))

        net = traingd(net, (P, T), (None, None), epochs, learning_rate = 1, block_size = 0, error_module = cox_error)
    except FloatingPointError:
        print('Aaawww....')
    outputs = net.sim(P)
    c_index = get_C_index(T, outputs)
    logger.info("C index = " + str(c_index))

    plot_network_weights(net)

    return net

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    glogger.setLoggingLevel(glogger.debug)

    #Real data

    filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset.txt"
    P, T = parse_file(filename, targetcols = [4, 5], inputcols = [2, -4, -3, -2, -1], ignorerows = [0], normalize = True)

    #remove tail censored
    P, T = copy_without_tailcensored(P, T)

    #Fake data
    #productfunction_wn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/productfunction_many_with_noise.txt'

    #The training sample
    #with_noise = productfunction_wn

    #P, T_wn = parse_file(with_noise, targetcols = [4], inputcols = [0, 1, 2, 3], ignorecols = [], ignorerows = [], normalize = False)

    #Amount to censor
    #ratio = 0.50

    #T_wn = censor_rndtest(T_wn, ratio)

    #Training sample
    #T = T_wn
    #filename = with_noise

    #Limit to incourage overtraining!
    #rows = sample(range(len(T)), 500)
    #P = P[rows]
    #T = T[rows]

    #Network part

    p = len(P[0]) #number of input covariates

    net = build_feedforward(p, 10, 1, output_function = 'linear')

    #Initial state
    outputs = net.sim(P)
    orderscatter(outputs, T, filename, 's')

    for var in xrange(len(P[0, :])):
        plot_input(P[:, var])

    glogger.show()

    epochs = 100

    for times in range(100):
        net = test(net, P, T, filename, epochs)

        outputs = net.sim(P)
        orderscatter(outputs, T, filename, 'o')
        raw_input("Press enter to show plots...")
        glogger.show()
