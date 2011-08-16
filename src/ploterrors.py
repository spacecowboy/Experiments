'''
Created on Aug 16, 2011

@author: jonask

Plots cross validation errors with error bars. The netsize on the bottom is the order of the files!!!
'''
import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print('Proper usage is: ' + sys.argv[0] + ' filename1 [filename2] [filename3] etc')
    sys.exit()

errors = {}
validations = {}
error_avg = {}
validation_avg = {}

fig = plt.figure()
ax = fig.add_subplot(111)
ps = []
labels = []

netsize = 0
for filename in sys.argv[1:]:
    state = 'None'
    netsize += 1
    with open(filename) as FILE:
        errors[filename] = np.array([], dtype = np.float64)
        validations[filename] = np.array([], dtype = np.float64)
        error_avg = {}
        validation_avg = {}
        for line in FILE:
            if line.startswith('Test Errors, Validation Errors'):
                state = 'first'
                continue
            if line.startswith('Test average, Validation average'):
                state = 'avg'
                continue

            if state == 'first':
                try:
                    vals = line.split(', ')
                    print(vals)
                    errors[filename] = np.append(errors[filename], float(vals[0]))
                    validations[filename] = np.append(validations[filename], float(vals[1]))
                except ValueError:
                    print 'Data ended, ending state...'
                    state = 'None'

            if state == 'avg':
                try:
                    print(vals)
                    vals = line.split(', ')
                    error_avg[filename] = float(vals[0])
                    validation_avg[filename] = float(vals[1])
                except ValueError:
                    print 'Data ended, ending state...'
                    state = 'None'

    ps.append(ax.errorbar(netsize - 0.1, error_avg[filename],
                 yerr = [[min(errors[filename])], [max(errors[filename])]],
                 marker = 'o',
                 color = 'k',
                 ecolor = 'k',
                 markerfacecolor = 'g',
                 label = filename + ' error',
                 capsize = 5,
                 linestyle = 'None'))

    labels.append(filename + ' error')

    ps.append(ax.errorbar(netsize + 0.1, validation_avg[filename],
                 yerr = [[min(validations[filename])], [max(validations[filename])]],
                 marker = 's',
                 color = 'k',
                 ecolor = 'r',
                 markerfacecolor = 'r',
                 label = filename + ' validation',
                 capsize = 5,
                 linestyle = 'None'))

    labels.append(filename + ' validation')

leg = fig.legend(ps, labels, 'lower right')

ax.set_xlabel("Number of hidden nodes")
ax.set_ylabel("Training and Validation errors")
ax.set_title('Cross validation results')

plt.xlim(0, 6)
plt.show()
