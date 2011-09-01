'''
Created on Aug 30, 2011

@author: jonask
'''
import sys
import os
from kalderstam.util.filehandling import read_data_file, parse_data
from survival.cox_error_in_c import get_C_index
from survival.plotting import kaplanmeier
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None #This makes matplotlib optional

def survival_stat(filename, thresholds = None):
    data = np.array(read_data_file(filename, ","))
    D, t = parse_data(data, inputcols = (2, 3, 4, 5, 6, 7, 8, 9, 10), ignorerows = [0], normalize = False)

    T = D[:, (2, 3)]
    outputs = D[:, (-1, 3)]
    C = get_C_index(T, outputs)

    print("C-index: " + str(C))
    print("Genetic error: " + str(1 / C))

    th = kaplanmeier(D, 2, 3, -1, threshold = thresholds)
    print("Threshold dividing the set in two equal pieces: " + str(th))
    if plt:
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        print("Need a valid filename! Exiting...")
        sys.exit()
    else:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        thresholds = [float(x) for x in sys.argv[2:]]
    else:
        thresholds = None
    survival_stat(filename, thresholds)
