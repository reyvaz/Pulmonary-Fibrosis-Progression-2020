import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib as mpl

def config_checkpoint(filepath = 'weights.h5', monitor ='val_loss'):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath = filepath,
        monitor = monitor,
        mode = 'min',
        save_best_only = True,
        save_weights_only=False,
        verbose = 0)
    return checkpoint

checkpoint = config_checkpoint()

best_results = [None]
class show_metrics(tf.keras.callbacks.Callback):
    def __init__(self, freq=None, watch='val_loss', mode = 'min'):
        self.freq = freq
        self.watch = watch
        self.mode = mode

    def on_train_begin(self, logs=None):
        if self.mode == 'min':
            self.best = np.Inf
            self.compare = tf.math.less
        else:
            self.best = -np.Inf
            self.compare = tf.math.greater
        self.best_logs = None
        self.best_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.watch)
        if self.compare(current, self.best):
            print('New best at Epoch {:03d} {} improved from {:.4f} to {:.4f}'.format(
                epoch,self.watch, self.best, current))
            self.best = current
            self.best_logs = logs
            self.best_epoch = epoch

        if self.freq:
            if epoch % self.freq == 0:
                items = ['{}: {:.4f}'.format(i[0], i[1]) for i in logs.items()]
                print('Epoch: {:03d}'.format(epoch), *items)

    def on_train_end(self, logs=None):
        items = ['{}: {:.4f}'.format(i[0], i[1]) for i in self.best_logs.items()]
        print('\nBest at Epoch: {:03d}'.format(self.best_epoch), *items)
        best_results[0] = items

def get_optimizer(opt, lr):
    LEARNING_RATE = lr
    optimizers = {'Nadam': tf.keras.optimizers.Nadam(learning_rate=LEARNING_RATE),
                  'Radam': tfa.optimizers.RectifiedAdam(learning_rate=LEARNING_RATE),
                  'Adam': tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  'SGD': tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)}
    return optimizers[opt]

axes_color = '#999999'
mpl.rcParams.update({'text.color' : "#999999", 'axes.labelcolor' : axes_color,
                     'font.size': 10, 'xtick.color':axes_color,'ytick.color':axes_color,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': axes_color, 'axes.linewidth':1.0, 'figure.figsize':[8, 4]})

def describe_ds(ds):
    print(str(ds).replace('<', '').replace('>', ''))
