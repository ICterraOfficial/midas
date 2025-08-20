import unittest

import numpy as np
import pandas as pd

from data import AdversarialSampler

class TestAdversarialSampler(unittest.TestCase):

    def setUp(self):
        data = {
            'Domain': ['Source', 'Source', 'Source', 'Source', 'Target', 'Target'] * 5,
            'BreastID': ['A', 'B', 'C', 'D', 'E', 'F'] * 5
        }
        self.domain_labels = pd.DataFrame(data)
        self.sampler = AdversarialSampler(self.domain_labels, batch_size=4, shuffle=False)

    def test_initialization(self):
        self.assertEqual(self.sampler.batch_size, 4)
        self.assertEqual(self.sampler.shuffle, False)
        self.assertEqual(len(self.sampler.source_domain_idx), 20)
        self.assertEqual(len(self.sampler.target_domain_idx), 20)

    def test_len(self):
        self.assertEqual(len(self.sampler), 10)

    def test_pad_indexes(self):
        indexes = [[1, 2, 3, 4, 5]]
        self.sampler._pad_indexes(indexes)
        self.assertEqual(len(indexes[0]), 8)  # Because batch size is 4, it should pad to 8

    def test_equalize_samples(self):
        source_data = self.domain_labels[self.domain_labels.Domain == 'Source']
        target_data = self.domain_labels[self.domain_labels.Domain == 'Target']
        equalized_samples = self.sampler._equalize_samples(source_data, target_data)
        self.assertEqual(len(equalized_samples), len(source_data))

    def test_iter(self):
        batches = list(iter(self.sampler))
        self.assertEqual(len(batches), 10)
        for batch in batches:
            self.assertEqual(len(batch), 4)

    def test_shuffle(self):
        sampler_shuffled = AdversarialSampler(self.domain_labels, batch_size=4, shuffle=True)
        batches_shuffled = list(iter(sampler_shuffled))
        self.assertEqual(len(batches_shuffled), 10)

        shuffled = []

        for batch in batches_shuffled:
            if batch == sorted(batch):
                shuffled.append(False)
            else:
                shuffled.append(True)

        self.assertEqual(any(shuffled), True)


if __name__ == '__main__':
    unittest.main()
