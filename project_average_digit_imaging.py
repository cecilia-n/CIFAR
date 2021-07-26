import pandas as pd
allTrainingData = pd.read_csv('./train.csv')
import numpy as np
import matplotlib.pyplot as plt

def brute_sum(thing, start=0):
    '''
    Function that adds all the elements of an iterable to the start value.

    parameters:
    thing: any iterable
    start: the value to which everything will be added

    returns:
    the sum
    '''
    for elem in thing:
        start += elem
    return start

# A script that plots all the averages.
fig, axs = plt.subplots(2, 5, figsize=(28, 28))
for digit, frequency in allTrainingData['label'].value_counts().iteritems():
    inds = allTrainingData['label'] == digit
    average = brute_sum(allTrainingData[inds].drop(columns='label').to_numpy(), start=np.zeros(784)) / frequency
    axs[divmod(digit, 5)].imshow(average.reshape(28, 28), cmap='gray')
plt.show()
