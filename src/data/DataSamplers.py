import math
import random

import numpy as np
import pandas as pd


class GMICSampler:
    """
    A custom batch sampler designed for unbalanced datasets, utilizing the strategy
    outlined in the GMIC paper. This sampler selects sample indexes based on their class distribution,
    allocating the number of samples for the negative, benign, and malignant classes as follows:
    (batch_size/2, batch_size/4, batch_size/4) respectively.

    See details: https://doi.org/10.1016%2Fj.media.2020.101908.

    Parameters
    ----------
    labels : pandas.Series
        The labels of the dataset.
    batch_size : int, optional
        The batch size. Default is 4.
    shuffle : bool, optional
        Whether to shuffle the indices. Default is True.

    Raises
    ------
    AssertionError
        If batch_size is less than 4 or not a multiple of 4.

    Attributes
    ----------
    labels : pandas.Series
        The labels of the dataset.
    batch_size : int
        The batch size.
    shuffle : bool
        Whether to shuffle the indices.
    """

    def __init__(self, labels, batch_size=4, shuffle=True):
        # The minimum size of the batch_size must be 4 because a batch must contain at least 1 benign
        # and 1 malign sample as well as 2 negative sample.
        assert batch_size >= 4 and batch_size % 4 == 0, 'Batch size must be greater than 4 and a multiple of 4.'
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Class labels.
        negative_label = '[0, 0]'
        benign_label = '[1, 0]'
        malign_label = '[0, 1]'

        # Find indexes for different classes.
        self.index_negative = labels[labels == negative_label].index.tolist()
        self.index_benign = labels[labels == benign_label].index.tolist()
        self.index_malign = labels[labels == malign_label].index.tolist()

        if self.shuffle:
            self._shuffle_indexes([self.index_negative, self.index_benign, self.index_malign])

        self._pad_indexes([self.index_negative, self.index_benign, self.index_malign])

        # Determine the minimum number of samples among all classes to
        # ensure batch indexes only go up to the length of the smallest class.
        self.n_min_class = min([len(self.index_negative), len(self.index_benign), len(self.index_malign)])

    def __len__(self):
        return self.n_min_class//(self.batch_size//4)

    def _shuffle_indexes(self, indexes):
        for index in indexes:
            random.shuffle(index)

    def _pad_indexes(self, indices):
        """
        Pad the given list of indexes with the last items to ensure each sublist has a length
        that is a multiple of the batch size. This prevents raising exceptions at the end of
        the loop when iterating over batches.

        Parameters
        ----------
        indices : list of lists
            The list of indices, where each sublist represents a batch of indices.

        Returns
        -------
        None
            The original list of indexes is modified in place.
        """

        # ToDo: The indices added should be lenght of batch_size - indices
        for index in indices:
            remainder = len(index) % self.batch_size
            n_copy_indices = self.batch_size - remainder

            if remainder != 0:
                index.extend(random.sample(index, n_copy_indices))

    def __iter__(self):
        """
        Iterates over batches of indexes.

        Yields
        ------
        list
            A batch of shuffled indices.
        """

        batch_idx = []

        for i in range(0, self.n_min_class, self.batch_size // 4):
            batch_idx_negative = self.index_negative[
                                 i * (self.batch_size // 2):i * (self.batch_size // 2) + (self.batch_size // 2)]
            batch_idx.extend(batch_idx_negative)

            batch_idx_benign = self.index_benign[i:i + (self.batch_size // 4)]
            batch_idx.extend(batch_idx_benign)

            batch_idx_malign = self.index_malign[i:i + (self.batch_size // 4)]
            batch_idx.extend(batch_idx_malign)

            random.shuffle(batch_idx)

            yield batch_idx
            batch_idx = []


class AdversarialSampler:
    """
    A sampler for domain adversarial training.

    Parameters
    ----------
    dataset : Dataset
        A Dataset instance. The dataset must have a `metadata` attribute containing
    batch_size : int, optional
        The size of the batch to sample. Must be a multiple of 4 and greater than or equal to 4. Default is 4.
    shuffle : bool, optional
        Whether to shuffle the domain labels before sampling. Default is True.
    """

    def __init__(self, dataset, batch_size=4, shuffle=True):
        """
        Initialize the AdversarialSampler.

        Parameters
        ----------
        dataset : Dataset
            A Dataset instance. The dataset must have a `metadata` attribute containing
            `Domain` and `BreastID` columns.
        batch_size : int, optional
            The size of the batch to sample. Must be a multiple of 4 and greater than or equal to 4. Default is 4.
        shuffle : bool, optional
            Whether to shuffle the domain labels before sampling. Default is True.
        """

        assert batch_size >= 4 and batch_size % 4 == 0, 'Batch size must be greater than 4 and a multiple of 4.'
        assert 'Domain' in dataset.metadata.columns, '`Domain` column must be present.'
        assert 'BreastID' in dataset.metadata.columns, '`BreastID` column must be present.'

        self.batch_size = batch_size
        self.shuffle = shuffle

        # Get only BreastID and Domain columns.
        domain_labels = dataset.metadata[['BreastID', 'Domain']].copy()

        # Decompose source and target data.
        source_domain = domain_labels[domain_labels.Domain=='Source']
        target_domain = domain_labels[domain_labels.Domain=='Target']

        if len(source_domain) > len(target_domain):
            # Equalize number of target samples to number of source samples.
            target_domain = self._equalize_samples(source_domain, target_domain)

        self.source_domain_idx = source_domain.index.tolist()
        self.target_domain_idx = target_domain.index.tolist()

        self._pad_indexes([self.source_domain_idx, self.target_domain_idx])

        self.step_size = self.batch_size // 2

    def __len__(self):
        return math.ceil(len(self.source_domain_idx) / self.step_size)

    def _pad_indexes(self, indexes):
        """
        Pad the given list of indexes with the last items to ensure each sublist has a length
        that is a multiple of the batch size. This prevents raising exceptions at the end of
        the loop when iterating over batches.

        Parameters
        ----------
        indexes : list of lists
            The list of indexes, where each sublist represents a batch of indexes.

        Returns
        -------
        None
            The original list of indexes is modified in place.
        """

        for index in indexes:
            remainder = len(index) % self.batch_size
            if remainder != 0:
                pad_size = self.batch_size - remainder
                index.extend(index[-pad_size:])

    def _equalize_samples(self, source_domain, target_domain):
        """Equalize number of source samples to number of target samples. If number of samples in target data is less
        than number of samples in source data, oversampling is performed.

        Parameters
        ----------
        source_domain: pd.Series
        target_domain: pd.Series

        Returns
        -------
        pd.Series
        """

        # First, find unique breast_ids. In this research, evaluation is performed based on two images and processing
        # both images (CC and MLO) of same breast is crucial.
        source_idx = source_domain.BreastID.unique()
        target_idx = target_domain.BreastID.unique()

        # Create same entries randomly to equalize numbers.
        target_idx = np.random.choice(target_idx, len(source_idx), replace=True)
        oversampled_data = pd.concat([target_domain[target_domain.BreastID == idx] for idx in target_idx])

        return oversampled_data

    def __iter__(self):
        """
        Iterates over batches of indexes.

        Yields
        ------
        list
            A batch of shuffled indices.
        """

        batch_idx = []

        if self.shuffle:
            # ! Do not reset indexes, they will be shuffled and passed to Dataloader.
            random.shuffle(self.source_domain_idx)
            random.shuffle(self.target_domain_idx)

        for i in range(0, len(self.source_domain_idx), self.step_size):
            batch_idx_source = self.source_domain_idx[i:i + self.step_size]
            batch_idx.extend(batch_idx_source)

            batch_idx_target = self.target_domain_idx[i:i + self.step_size]
            batch_idx.extend(batch_idx_target)
            random.shuffle(batch_idx)
            yield batch_idx
            batch_idx = []


class TargetBalancedSampler:
    """
    A BatchSampler that creates balanced batches containing an equal number of samples
        from a target class and a sampled class.

        Parameters
        ----------
        metadata : pd.DataFrame
            A DataFrame containing metadata with at least one column for OneHotLabel.
        target_class : str
            The target class label from which to sample half of the batch.
            Target class must be in onehot format.
            Examples: '[0, 1]'
        batch_size : int, optional
            The total number of samples in each batch (default is 4).
        shuffle : bool, optional
            Whether to shuffle the indices of the classes before creating batches (default is True).

        Raises
        ------
        AssertionError
            If `batch_size` is less than 2 or not a multiple of 2.

        Methods
        -------
        __len__() -> int
            Returns the number of batches that can be created.
        __iter__() -> Iterator[list]
            Iterates over batches of indices.
        _shuffle_indexes(onehot_labels: dict) -> dict
            Shuffles the indices of the one-hot labels.
        _pad_indexes(onehot_labels: dict) -> None
            Pads the indices of the one-hot labels to ensure each sublist has a length
            that is a multiple of the batch size.
    """

    def __init__(self, metadata, target_class, batch_size=4, shuffle=True, padding=False):
        """
        Initializes the TargetBalancedSampler.

        Parameters
        ----------
        metadata : pd.DataFrame
            A DataFrame containing metadata with at least one column named 'OneHotLabel',
            representing one-hot encoded labels of the dataset.
        target_class : str
            The target class label (in one-hot encoded format, e.g., '[0, 1]') from which
            half of each batch will be drawn.
        batch_size : int, optional
            The total number of samples in each batch (must be an even number, default is 4).
        shuffle : bool, optional
            If True, the indices for both the target and sampled classes will be shuffled before
            batching (default is True).
        padding : bool, optional
            If True, ensures the number of samples for each class is padded to be a multiple of
            half the batch size, allowing for even sampling in each batch (default is False).

        Raises
        ------
        AssertionError
            If the `batch_size` is less than 2 or not a multiple of 2.
        """

        assert batch_size >= 2 and batch_size % 2 == 0, 'Batch size must be greater than 2 and a multiple of 2.'
        self.batch_size = batch_size

        onehot_labels = metadata.groupby('OneHotLabel').groups

        if shuffle:
            onehot_labels = self._shuffle_indexes(onehot_labels)
        if padding:
            onehot_labels = self._pad_indices(onehot_labels)

        self.target_class_indices = onehot_labels.pop(target_class)
        self.sampled_class_indices = list(onehot_labels.values())[0]

        self.n_target_class = len(self.target_class_indices)

    def __len__(self):
        """
        Returns the number of batches that can be created.

        Returns
        -------
        int
            The number of batches based on the target class size and the batch size.
        """
        return np.ceil(self.n_target_class*2 / self.batch_size).astype(int)

    def _shuffle_indexes(self, onehot_labels):
        for labels, indices in onehot_labels.items():
            if type(indices) != list:
                indices = indices.tolist()
            random.shuffle(indices)
            onehot_labels[labels] = indices

        return onehot_labels

    def _pad_indices(self, onehot_labels):
        """
        Pad the given list of indexes with the last items to ensure each sublist has a length
        that is a multiple of the batch size. This prevents raising exceptions at the end of
        the loop when iterating over batches.

        Parameters
        ----------
        indexes : list of lists
            The list of indexes, where each sublist represents a batch of indexes.

        Returns
        -------
        None
            The original list of indexes is modified in place.
        """

        for label, indices in onehot_labels.items():
            if type(indices) != list:
                indices = indices.tolist()

            remainder = len(indices) % self.batch_size
            n_copy_indices = self.batch_size - remainder

            if remainder != 0:
                indices.extend(random.sample(indices, n_copy_indices))

            onehot_labels[label] = indices

        return onehot_labels

    def __iter__(self):
        """
        Iterates over batches of indexes.

        Yields
        ------
        list
            A batch of shuffled indices.
        """

        batch_idx = []
        step_size = self.batch_size // 2

        for i in range(0, self.n_target_class, step_size):
            batch_idx_target = self.target_class_indices[i:i + step_size]
            batch_idx.extend(batch_idx_target)

            batch_idx_sampled = self.sampled_class_indices[i:i + step_size]
            batch_idx.extend(batch_idx_sampled)

            random.shuffle(batch_idx)
            yield batch_idx
            batch_idx = []
