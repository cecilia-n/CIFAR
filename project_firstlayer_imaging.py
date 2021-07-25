import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import ceil

import project_sklearn
from project_sklearn import mlp

first_coefs = mlp.coefs_[0].T

first_layer_size = len(first_coefs)

fig, axs = plt.subplots(ceil(len(first_coefs)/10), 10, figsize=(28, 28))
scaler = MinMaxScaler()

if first_layer_size > 10:
    for num, coefs in enumerate(first_coefs):
        new_coefs = scaler.fit_transform(coefs.reshape(-1, 1))
        axs[divmod(num, 10)].imshow(new_coefs.reshape(28, 28), cmap='gray')
else:
    for num, coefs in enumerate(first_coefs):
        new_coefs = scaler.fit_transform(coefs.reshape(-1, 1))
        axs[num].imshow(new_coefs.reshape(28, 28), cmap='gray')

plt.show()
