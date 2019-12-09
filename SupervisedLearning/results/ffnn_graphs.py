from matplotlib import pyplot
import numpy as np
import os

NUM_EPOCHS = 250
hidden_layer_sizes = [10, 20, 50, 70, 100]
handles = [None] * len(hidden_layer_sizes)
fig = pyplot.figure("CS 4033/5033 - Hennen/Bost")
ax = fig.add_subplot(111)
ax.set_ylabel('mse')
ax.set_xlabel('epoch')
ax.set_title('Cam\'s model (1-hidden)')

for idx_size, hidden_layer_size in enumerate(hidden_layer_sizes):
    fname = f'results_h{hidden_layer_size}_e{NUM_EPOCHS}.txt'
    if os.path.exists(fname):
        mse_data = np.loadtxt(fname, dtype=np.float32)
        pyplot.plot(mse_data, label=f"H={hidden_layer_size}")
pyplot.legend()
pyplot.show()