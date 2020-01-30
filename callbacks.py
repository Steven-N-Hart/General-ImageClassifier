import tensorflow as tf
import os


class CallBacks:

    def __init__(self, learning_rate=0.01, log_dir=None, optimizer=None):
        self.log_dir = log_dir
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.callbacks = self.get_callbacks()

    def _get_tb(self):
        return tf.keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                              histogram_freq=1,
                                              write_graph=True,
                                              update_freq='epoch',
                                              write_images=False)

    def _get_cp(self):
        return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.log_dir, 'cp-{epoch:04d}.ckpt'),
                                                  verbose=1,
                                                  save_weights_only=True,
                                                  save_frequency=1)


    @staticmethod
    def _get_es():
        return tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30)

    def get_callbacks(self):
        return [self._get_tb(), self._get_cp(), self._get_es()]
