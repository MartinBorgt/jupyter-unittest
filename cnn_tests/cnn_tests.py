"""
Demonstration of running unnit tests on functions in a .pynb file.
"""

import unittest
from collections import namedtuple
import tensorflow as tf

from setup_notebook_import import NotebookFinder

NotebookFinder.setup()

from cnn.cnn import (
    create_model,
    downsample,
    jaccard_bce_loss,
    jaccard_loss,
    load_pneumonia_locations,
    load_file_names,
    Generator,
    resblock,
)


class DataLocationsMixin:
    train_labels_path = "../input/stage_2_train_labels.csv"
    train_images_path = "../input/stage_2_train_images"


class LoadDataTestCase(DataLocationsMixin, unittest.TestCase):
    def test_load_pneumonia_locations(self):
        pneumonia_locations = load_pneumonia_locations(self.train_labels_path)
        self.assertTrue(len(pneumonia_locations))

    def test_load_file_names(self):
        ratio = 0.7
        train_filenames, validation_filenames = load_file_names(
            path=self.train_images_path, train_ratio=ratio
        )
        self.assertGreater(
            1,
            abs(len(train_filenames) * (1 - ratio) - len(validation_filenames) * ratio),
        )
        train_set = set(train_filenames)
        val_set = set(validation_filenames)
        self.assertFalse(train_set.intersection(val_set))


class GeneratorTestCase(DataLocationsMixin, unittest.TestCase):
    def setUp(self):
        super(GeneratorTestCase, self).setUp()
        self.pneumonia_locations = load_pneumonia_locations(self.train_labels_path)
        self.pneumonia_filenames = {
            "{}.dcm".format(key) for key in self.pneumonia_locations
        }
        self.batch_size = 32
        self.train_filenames, validation_filenames = load_file_names(
            path=self.train_images_path, train_ratio=0.1
        )
        self.generator = Generator(
            self.train_images_path,
            self.train_filenames,
            self.pneumonia_locations,
            batch_size=self.batch_size,
            image_size=256,
            shuffle=True,
            predict=False,
        )

    def test_len(self):
        self.assertEqual(len(self.train_filenames) // 32, len(self.generator))

    def test_mask(self):
        def image_mask_shows_pneumonia(filename):
            _img, mask = self.generator.__load__(filename)
            return any([any(row) for row in mask])

        no_pneumonia_filenames = set(self.train_filenames).difference(
            self.pneumonia_filenames
        )

        self.assertFalse(image_mask_shows_pneumonia(no_pneumonia_filenames.pop()))
        self.assertTrue(image_mask_shows_pneumonia(self.pneumonia_filenames.pop()))

    def test_load(self):
        pneumonia_filename = self.pneumonia_filenames.pop()
        img, mask = self.generator.__load__(pneumonia_filename)
        pred_img = self.generator.__loadpredict__(pneumonia_filename)
        self.assertTrue(
            all(
                [
                    a == b
                    for row, pred_row in zip(img, pred_img)
                    for a, b in zip(row, pred_row)
                ]
            )
        )

    def test_getitem(self):
        self.assertEqual(len(self.generator[0][0]), self.batch_size)


class NetworkTestCase(unittest.TestCase):
    def setUp(self):
        super(NetworkTestCase, self).setUp()
        self.in_size = 256
        self.n_outputs = 4
        self.inputs = tf.keras.Input(shape=(self.in_size, self.in_size, 1))

    def test_downsample(self):
        outputs = downsample(self.n_outputs, self.inputs)
        self.assertEqual(
            outputs.shape.as_list(),
            [None, self.in_size / 2, self.in_size / 2, self.n_outputs],
        )

    def test_resblock(self):
        outputs = resblock(self.n_outputs, self.inputs)
        self.assertEqual(
            outputs.shape.as_list(), [None, self.in_size, self.in_size, self.n_outputs]
        )

    def test_create_model(self):
        model = create_model(self.in_size, self.n_outputs)
        self.assertEqual(
            model.inputs[0].shape.as_list(), [None, self.in_size, self.in_size, 1]
        )
        self.assertEqual(
            model.outputs[0].shape.as_list(), [None, self.in_size, self.in_size, 1]
        )


class LossTestCase(unittest.TestCase):
    def setUp(self):
        super(LossTestCase, self).setUp()

        self.sess = tf.Session()

        self.place_holder_true = tf.placeholder(dtype=tf.float32, shape=(None, None))
        self.place_holder_pred = tf.placeholder(dtype=tf.float32, shape=(None, None))

        y_true = [[1, 1, 1], [1, 0, 0], [0, 0, 0]]
        y_pred = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]

        perfect_y_true = [[1, 1, 1], [1, 0, 0], [0, 0, 0]]
        perfect_y_pred = [[1, 1, 1], [1, 0, 0], [0, 0, 0]]

        none_y_true = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
        none_y_pred = [[0, 0, 0], [0, 0, 1], [0, 1, 1]]
        Example = namedtuple("Example", ("true", "predicted"))
        self.examples = {
            "prediction": Example(y_true, y_pred),
            "perfect prediction": Example(perfect_y_true, perfect_y_pred),
            "no overlap prediction": Example(none_y_true, none_y_pred),
        }

    def _get_session_output(self, function, example):
        loss = function(self.place_holder_true, self.place_holder_pred)
        return self.sess.run(
            loss,
            feed_dict={
                self.place_holder_true: self.examples[example].true,
                self.place_holder_pred: self.examples[example].predicted,
            },
        )

    def test_jaccard_loss(self):
        self.assertEqual(self._get_session_output(jaccard_loss, "prediction"), 0.25)
        self.assertEqual(
            self._get_session_output(jaccard_loss, "perfect prediction"), 0
        )
        self.assertEqual(
            self._get_session_output(jaccard_loss, "no overlap prediction"), 1
        )

    def test_iou_bce_loss(self):
        self.assertTrue(
            self._get_session_output(jaccard_bce_loss, "perfect prediction") < 0.0001
        )
        self.assertNotEqual(
            self._get_session_output(jaccard_bce_loss, "no overlap prediction"), 1
        )


if __name__ == "__main__":
    unittest.main()
