import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import ceil

import project_sklearn
from project_sklearn import mlp

first_coefs = mlp.coefs_[0].T

first_layer_size = len(first_coefs)

# fig, axs = plt.subplots(ceil(len(first_coefs)/10), 10, figsize=(28, 28))
scaler = MinMaxScaler()

width = 10
split = 100

if first_layer_size > width:
    for num, coefs in enumerate(first_coefs):
        new_num = num % split
        if not new_num:
            if num:
                plt.show()
            input("Press ENTER to continue")
            fig, axs = plt.subplots(ceil(split/width), width, figsize=(28, 28))
        new_coefs = scaler.fit_transform(coefs.reshape(-1, 1))
        axs[divmod(new_num, width)].imshow(new_coefs.reshape(28, 28), cmap='gray')
    plt.show()
else:
    fig, axs = plt.subplots(first_layer_size, figsize=(28, 28))
    for num, coefs in enumerate(first_coefs):
        new_coefs = scaler.fit_transform(coefs.reshape(-1, 1))
        axs[num].imshow(new_coefs.reshape(28, 28), cmap='gray')
    plt.show()
