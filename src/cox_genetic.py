from kalderstam.util.filehandling import parse_file, load_network, save_network, \
    get_validation_set, print_output
from kalderstam.neural.network import build_feedforward, \
    build_feedforward_multilayered, build_feedforward_committee
import numpy
from survival.cox_error import get_C_index
from survival.cox_genetic import c_index_error
import survival.cox_error as cox_error
from survival.cox_error import censor_rndtest, pre_loop_func, calc_sigma, calc_beta, cox_error as cerror
import logging
from kalderstam.neural.training.gradientdescent import traingd
from kalderstam.neural.training.davis_genetic import train_evolutionary
#from kalderstam.neural.training.genetic import train_evolutionary
import kalderstam.util.graphlogger as glogger
try:
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab
    from kalderstam.matlab.matlab_functions import plot_network_weights
except ImportError:
    plt = None
except RuntimeError:
    plt = None

from random import sample
from kalderstam.neural.training.committee import train_committee
import sys
logger = logging.getLogger('kalderstam.neural.cox_training')

def beta_error(target, result):
    vars = pre_loop_func(None, None, target, 0)
    sigma = calc_sigma(result)
    beta, beta_risk, part_func, weighted_avg = calc_beta(result, vars['timeslots'], vars['risk_groups'])
    return len(result) * cerror(beta, sigma)

def plot_input(data):
    if plt:
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

def test(net, P, T, vP, vT, filename, epochs, mutation_rate = 0.05, population_size = 50):
    logger.info("Running genetic test for: " + filename + ' ' + str(epochs))
    print("\nTraining set:")
    print("Number of patients with events: " + str(T[:, 1].sum()))
    print("Number of censored patients: " + str((1 - T[:, 1]).sum()))
    print("\nValidation set:")
    print("Number of patients with events: " + str(vT[:, 1].sum()))
    print("Number of censored patients: " + str((1 - vT[:, 1]).sum()))


    outputs = net.sim(P)
    c_index = get_C_index(T, outputs)
    logger.info("C index = " + str(c_index))

    try:
        net = train_evolutionary(net, (P, T), (vP, vT), epochs, error_function = c_index_error, population_size = population_size, mutation_chance = mutation_rate)

        outputs = net.sim(P)
        c_index = get_C_index(T, outputs)
        logger.info("C index = " + str(c_index))

        #net = traingd(net, (P, T), (None, None), epochs * 2, learning_rate = 1, block_size = 0, error_module = cox_error)
    except FloatingPointError:
        print('Aaawww....')
    outputs = net.sim(P)
    c_index = get_C_index(T, outputs)
    logger.info("C index test = " + str(c_index))

    outputs = net.sim(vP)
    c_index = get_C_index(vT, outputs)
    logger.info("C index vald = " + str(c_index))

    if plt:
        plot_network_weights(net)

    return net

def train_single():
    try:
        netsize = input('Number of hidden nodes? [3]: ')
    except SyntaxError as e:
        netsize = 3

    try:
        pop_size = input('Population size? [50]: ')
    except SyntaxError as e:
        pop_size = 50

    try:
        mutation_rate = input('Please input a mutation rate (0.25): ')
    except SyntaxError as e:
        mutation_rate = 0.25

    SB22 = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset_SB22.txt"
    Benmargskohorten = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset_Benmargskohorten.txt"
    SB91b = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset_SB91b.txt"
    all_studies = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset.txt"

    #Real data
    print("Studies to choose from:")
    print("1: SB22")
    print("2: Benmargskohorten")
    print("3: SB91b")
    print("0: All combined (default)")

    try:
        study = input("Which study to train on? [0]: ")
    except SyntaxError as e:
        study = 0

    if study == 1:
        filename = SB22
    elif study == 2:
        filename = Benmargskohorten
    elif study == 3:
        filename = SB91b
    else:
        filename = all_studies

    try:
        columns = input("Which columns to include? (Do NOT forget trailing comma if only one column is used, e.g. '3,'\nAvailable columns are: 2, -4, -3, -2, -1. Just press ENTER for all columns.\n")
    except SyntaxError:
        columns = (2, -4, -3, -2, -1)
    #P, T = parse_file(filename, targetcols = [4, 5], inputcols = [2, -4, -3, -2, -1], ignorerows = [0], normalize = True)
    P, T = parse_file(filename, targetcols = [4, 5], inputcols = columns, ignorerows = [0], normalize = True)

    #Used for output comparison
    studies = {}
    studies[SB22] = parse_file(SB22, targetcols = [4, 5], inputcols = columns, ignorerows = [0], normalize = True)
    studies[Benmargskohorten] = parse_file(Benmargskohorten, targetcols = [4, 5], inputcols = columns, ignorerows = [0], normalize = True)
    studies[SB91b] = parse_file(SB91b, targetcols = [4, 5], inputcols = columns, ignorerows = [0], normalize = True)
    studies[all_studies] = parse_file(all_studies, targetcols = [4, 5], inputcols = columns, ignorerows = [0], normalize = True)

    #remove tail censored
    P, T = copy_without_tailcensored(P, T)

    #Divide into validation sets
    ((tP, tT), (vP, vT)) = get_validation_set(P, T, validation_size = 0.25)

    #Network part

    p = len(P[0]) #number of input covariates

    net = build_feedforward(p, netsize, 1, output_function = 'linear')
    #net = build_feedforward_multilayered(p, [7, 10], 1, output_function = 'linear')

    #Initial state
    #outputs = net.sim(tP)
    #orderscatter(outputs, tT, filename, 's')

    for var in xrange(len(P[0, :])):
        try:
            plot_input(tP[:, var])
        except FloatingPointError as e:
            logger.error('Var ' + str(var) + ' failed plotting somehow...')
            print(e)

    glogger.show()

    try:
        epochs = input("Number of generations (200): ")
    except SyntaxError as e:
        epochs = 200

    for times in range(100):
        #train
        net = test(net, tP, tT, vP, vT, filename, epochs, population_size = pop_size, mutation_rate = mutation_rate)

        raw_input("Press enter to show plots...")
        glogger.show()
        try:
            answer = input('Do you wish to print network output? [y]: ')
        except SyntaxError as e:
            answer = 'y'
        if answer == 'y' or answer == 'yes':
            ps, ts = studies[filename]
            outputs = net.sim(ps)
            print_output(filename, outputs)

def cross_validation_test():
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

    print("\nData set:")
    print("Number of patients with events: " + str(T[:, 1].sum()))
    print("Number of censored patients: " + str((1 - T[:, 1]).sum()))

    #try:
    #    comsize = input("Number of networks to cross-validate [10]: ")
    #except SyntaxError:
    comsize = 4
    print('\nNumber of networks to cross-validate: ' + str(comsize))

    times_to_cross = 3
    print('\nNumber of times to repeat cross-validation: ' + str(times_to_cross))

    #try:
    #    netsize = input('Number of hidden nodes [3]: ')
    #except SyntaxError as e:
    if len(sys.argv) < 2:
        netsize = 3
    else:
        netsize = sys.argv[1]
    print("Number of hidden nodes: " + str(netsize))

    #try:
    #    pop_size = input('Population size [50]: ')
    #except SyntaxError as e:
    pop_size = 50
    print("Population size: " + str(pop_size))

    #try:
    #    mutation_rate = input('Please input a mutation rate (0.25): ')
    #except SyntaxError as e:
    mutation_rate = 0.15
    print("Mutation rate: " + str(mutation_rate))

    #try:
    #    epochs = input("Number of generations (200): ")
    #except SyntaxError as e:
    epochs = 200
    print("Epochs: " + str(epochs))

    for _ in xrange(times_to_cross):
        com = build_feedforward_committee(comsize, len(P[0]), netsize, 1, output_function = 'linear')

        #1 is the column in the target array which holds teh binary censoring information
        test_errors, vald_errors = train_committee(com, train_evolutionary, P, T, 1, epochs, error_function = c_index_error, population_size = pop_size, mutation_chance = mutation_rate)

        print('\nTest Errors, Validation Errors:')
        for terr, verr in zip(test_errors.values(), vald_errors.values()):
            print(str(terr) + ", " + str(verr))


if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    glogger.setLoggingLevel(glogger.debug)

    #train_single()
    cross_validation_test()
