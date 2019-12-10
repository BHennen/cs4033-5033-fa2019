from matplotlib import pyplot
import numpy as np
import os

# This is the script that created the graphs in the FNN section.
# At this point in time everything is hard-coded, but you
# can unzip results_cam_2hidden_momentum.zip and 
# results_keras_2hidden.zip then run this script to generate
# Figure 5

NUM_EPOCHS = 100
hidden_layer_sizes = [10, 20]#, 50, 70, 100]
fig = pyplot.figure("CS 4033/5033 - Hennen/Bost")
ax = fig.add_subplot(111)
ax.set_ylabel('accuracy')
ax.set_xlabel('epoch')
ax.set_title('Cam vs Keras (2-hidden, momentum)')

for idx_size, hidden_layer_size in enumerate(hidden_layer_sizes):
    fname = f'results_h{hidden_layer_size}_e{NUM_EPOCHS}.npz'
    fname_alt = f'results_h{hidden_layer_size}_e{NUM_EPOCHS}.txt'
    keras_fname = f'keras_h{hidden_layer_size}.npz'
    file_data = None
    mse_data = None
    # Load cam data
    if os.path.exists(fname):
        file_data = np.load(fname)
        mse_data = file_data['arr_0'] if file_data is not None else None
    elif os.path.exists(fname_alt):
        mse_data = np.loadtxt(fname_alt, dtype=np.float32)

    # Load Keras data
    if os.path.exists(keras_fname):
        keras_data = np.load(keras_fname)


    # Load metrics
    accuracies = file_data['arr_1'] if file_data is not None else None
    if mse_data is not None:
        mse_data = mse_data[0:100]
    keras_mse = keras_data['arr_0']
    keras_acc = keras_data['arr_1']

    # Currently hard-coded for mse_data
    pyplot.plot(mse_data, label=f"H={hidden_layer_size}")
    pyplot.plot(keras_mse, label=f"KerasH={hidden_layer_size}")
pyplot.legend(loc='upper right')
pyplot.show()
