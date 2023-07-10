import unittest
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from datascience_core.modelling.models.neural_network import NeuralNetworkBinaryClassifierPipeline
import shutil
import os


class TestNeuralNetworkBinaryClassifier(unittest.TestCase):
    """
    Creates a Neural Network model on some test data and checks that each of the custom functions runs as expected
    """
    def setUp(self) -> None:
        ds_split = tfds.load("penguins/processed", split=['train[20%:80%]', 'train[:20%]', 'train[80%:]'], batch_size=32,  as_supervised=True)
        self.figure_dir = 'test_figures'
        if os.path.exists(self.figure_dir):
            shutil.rmtree(self.figure_dir)
        ds_test = ds_split[2]
        ds_val = ds_split[1]
        ds_train = ds_split[0]
        assert isinstance(ds_test, tf.data.Dataset)

        inputs = tf.keras.Input(shape=(4,))
        x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
        x = tf.keras.layers.Dense(10, activation=tf.nn.relu)(x)
        outputs = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(x)
        
        model_wrapper = NeuralNetworkBinaryClassifierPipeline(inputs=inputs, outputs=outputs, figure_dir=self.figure_dir)
        model_wrapper.model.compile()
        model_wrapper.model.compile(
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
        )
        print(model_wrapper.model.summary())
        model_wrapper.fit(ds_train, epochs=2, steps_per_epoch=10, validation_data=ds_val)
        self.test_model = model_wrapper
        self.test = ds_test

    def test_predict(self):
        predictions = self.test_model.predict(self.test)
        assert predictions.shape == (67, 3)

    def test_predict_proba(self):
        predictions = self.test_model.predict_proba(self.test)
        assert predictions.shape == (67, 2, 3)

    def test_plot_and_save_history(self):
        self.test_model.plot_and_save_history()
        for key in ['loss', 'accuracy']:
            expected_path = os.path.join(os.getcwd(), f"{self.figure_dir}", f"{key}.png")
            print(os.listdir(os.path.join(os.getcwd(), f"{self.figure_dir}")))
            assert os.path.exists(expected_path), f"Expected path {expected_path} missing"




