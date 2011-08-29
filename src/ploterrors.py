'''
Created on Aug 16, 2011

@author: jonask

Plots cross validation errors with error bars. The netsize on the bottom is the order of the files!!!
'''
import sys
import numpy as np
import matplotlib.pyplot as plt

def reverse_error(e):
    ''' Reverse the error:
    1 / (C - 0.5) - 2
    1 / C
    '''
    #return 1.0 / (float(e) + 2.0) + 0.5
    return 1.0 / float(e)

if len(sys.argv) < 2:
    print('Proper usage is: ' + sys.argv[0] + ' filename1 [filename2] [filename3] etc')
    sys.exit()

errors = {}
validations = {}
error_avg = {}
validation_avg = {}
error_med = {}
validation_med = {}

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
                    errors[filename] = np.append(errors[filename], reverse_error(vals[0]))
                    validations[filename] = np.append(validations[filename], reverse_error(vals[1]))
                except ValueError:
                    print 'Data ended, ending state...'
                    state = 'None'

            if state == 'avg':
                try:
                    print(vals)
                    vals = line.split(', ')
                    #error_avg[filename] = reverse_error(vals[0])
                    #validation_avg[filename] = reverse_error(vals[1])
                except ValueError:
                    print 'Data ended, ending state...'
                    state = 'None'
        error_avg[filename] = errors[filename].mean()
        validation_avg[filename] = validations[filename].mean()
        error_med[filename] = np.median(errors[filename])
        validation_med[filename] = np.median(validations[filename])
        
        if len(errors[filename]) > 0:
            plotlines, caplines, barlinecols = ax.errorbar(netsize - 0.1, error_avg[filename],
                                         yerr = [[error_avg[filename] - min(errors[filename])], [-error_avg[filename] + max(errors[filename])]],
                                         marker = 'o',
                                         color = 'k',
                                         ecolor = 'k',
                                         markerfacecolor = 'g',
                                         label = filename + ' error',
                                         capsize = 5,
                                         linestyle = 'None')
            ps.append(plotlines)

            labels.append(filename + ' error')

        if len(validations[filename]) > 0:
            plotlines, caplines, barlinecols = ax.errorbar(netsize + 0.1, validation_avg[filename],
                                         yerr = [[validation_avg[filename] - min(validations[filename])], [-validation_avg[filename] + max(validations[filename])]],
                                         marker = 's',
                                         color = 'k',
                                         ecolor = 'r',
                                         markerfacecolor = 'r',
                                         label = filename + ' validation',
                                         capsize = 5,
                                         linestyle = 'None')
            ps.append(plotlines)

            for val_error in validations[filename]:
                ax.plot(netsize + 0.1, val_error, 'r+') 

            labels.append(filename + ' validation')

#leg = fig.legend(ps, labels, 'lower right')

ax.set_xlabel("Number of hidden nodes -->")
ax.set_ylabel("Training and Validation C-Index values -->")
ax.set_title('Cross validation C-Index results.')

plt.xlim(0, netsize + 1)
plt.show()
