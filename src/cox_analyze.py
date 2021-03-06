from kalderstam.matlab.matlab_functions import plot_network_weights
from kalderstam.util.filehandling import parse_file, load_network, save_network
from kalderstam.neural.network import build_feedforward
import numpy
from survival.cox_error import orderscatter, get_C_index
import survival.cox_error as cox_error
import kalderstam.util.graphlogger as glogger
import logging
from kalderstam.neural.training.gradientdescent import traingd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import sample
logger = logging.getLogger('kalderstam.neural.cox_training')

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

def test(net, P, T, filename, epochs, learning_rate, block_size):
    logger.info("Running test for: " + filename + ' ' + str(epochs) + ", rate: " + str(learning_rate) + ", block_size: " + str(block_size))
    print("Number of patients with events: " + str(T[:, 1].sum()))
    print("Number of censored patients: " + str((1 - T[:, 1]).sum()))

    outputs = net.sim(P)
    c_index = get_C_index(T, outputs)
    logger.info("C index = " + str(c_index))

    try:
        #net = train_cox(net, (P, T), (None, None), timeslots, epochs, learning_rate = learning_rate)
        net = traingd(net, (P, T), (None, None), epochs, learning_rate, block_size, error_module = cox_error)
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

    filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset.txt"
    P, T = parse_file(filename, targetcols = [4, 5], inputcols = [2, -4, -3, -2, -1], ignorerows = [0], normalize = True)
    #P, T = parse_file(filename, targetcols = [4, 5], inputcols = [2, -3], ignorerows = [0], normalize = True)

    #Remove tail censored
    P, T = copy_without_tailcensored(P, T)

    #Limit to incourage overtraining!
    #rows = sample(range(len(T)), 100)
    #P = P[rows]
    #T = T[rows]

    p = len(P[0]) #number of input covariates

    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/4x10x10x1.ann')
    net = build_feedforward(p, 30, 1, output_function = 'linear')

    #Initial state
    outputs = net.sim(P)
    orderscatter(outputs, T, filename, 's')

    for var in xrange(len(P[0, :])):
        plot_input(P[:, var])

    glogger.show()

    epochs = 20000
    rate = 1
    block_size = 0

    for times in range(100):
        net = test(net, P, T, filename, epochs, rate, block_size)

        outputs = net.sim(P)
        orderscatter(outputs, T, filename, 'o')
        raw_input("Press enter to show plots...")
        glogger.show()
