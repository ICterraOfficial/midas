from ast import literal_eval
import warnings
warnings.simplefilter("once", UserWarning)

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
import torch
from torch.utils.data import Dataset

__all__ = ['ClassificationDataset', 'SegmentationDataset', 'ObjectDetectionDataset']


class BaseDataset(Dataset):
    """
    Base class for mammography image datasets. This class provides functionality for reading metadata
    from Excel files, and applying transformations to images and segmentation masks.

    It supports both custom transformations as well as those from the official PyTorch and Albumentations libraries.
    """

    def __init__(self, metadata_path, transform=None):
        """
        Initializes the dataset with metadata and optional image transformations.

        Parameters
        ----------
        metadata_path : str
            Path to the Excel file containing DICOM paths and corresponding labels.

        transform : dict, optional
            Dictionary specifying image transformations to be applied. Should include:
                - 'dicom': list of custom transformations for DICOM images.
                - 'pytorch': torchvision.transforms.Compose object for standard PyTorch transformations.
        """

        self.metadata = pd.read_excel(metadata_path)

        breast_id_check = 'BreastID' in self.metadata.columns
        file_path_check = 'FilePath' or 'DICOM' in self.metadata.columns
        one_hot_label_check = 'OneHotLabel' in self.metadata.columns
        image_laterality_check = 'ImageLaterality' in self.metadata.columns

        assert (
                breast_id_check and file_path_check and one_hot_label_check and image_laterality_check
        ), (
            "Missing one or more required columns in the metadata file: "
            "'BreastID', 'FilePath', 'OneHotLabel', 'ImageLaterality'."
        )

        self.transform = transform
        if transform is not None:
            t_keys = self.transform.keys()
            assert ('dicom' in t_keys)|('A' in t_keys)|('pytorch' in t_keys), ("Transform key values "
                                        "are not matched with the 'dicom', 'A' (Albumentation), and 'pytorch' keys.")

        # Convert string-formatted one-hot labels from Excel (e.g., '[0, 1]') to a list of integers.
        self.metadata.OneHotLabel = self.metadata.OneHotLabel.apply(literal_eval)
        if 'DomainLabel' in self.metadata.keys():
            self.metadata.DomainLabel = self.metadata.DomainLabel.apply(literal_eval)

        # To ensure that loaded data indexes are correct.
        self.metadata.reset_index(drop=True, inplace=True)

    def __len__(self):
        return self.metadata.shape[0]

    def _apply_transform(self, image, mask=None, **kwargs):
        """
        Apply transformations to a mammography image.

        Parameters
        ----------
        image : numpy.ndarray
            Raw mammography image.
        dicom_data : pydicom.Dataset, optional
            Parsed DICOM metadata associated with the image.
        image_laterality : str, optional
            Indicates whether the image is from the left or right breast (e.g., 'L' or 'R').
        mask : numpy.ndarray, optional
            Segmentation masks corresponding to the image.

        Returns
        -------
        image : torch.Tensor or numpy.ndarray
            Transformed mammography image.
        """

        # Apply custom transformations that developed for mammography images.
        if 'dicom' in self.transform.keys() and self.transform['dicom']:
            for t in self.transform['dicom']:
                image, mask = t(image, mask=mask, coors=False, **kwargs)

        # Apply transformations of Albumentations.
        if 'A' in self.transform.keys() and self.transform['A'] is not None:
            mask = mask.numpy()
            mask_benign = mask[0]
            mask_malign = mask[1]
            transformed = self.transform['A'](image=image,
                                              mask_benign=mask_benign, mask_malign=mask_malign)
            image = transformed['image']
            mask_benign = transformed['mask_benign']
            mask_malign = transformed['mask_malign']
            mask = np.array([mask_benign, mask_malign])

        # Apply transformations of Torchvision. Torchvision transforms is only applicable for images not masks.
        if 'pytorch' in self.transform.keys() and self.transform['pytorch']:
            if mask is not None:
                warnings.warn("PyTorch transformation cannot be applied to the masks."
                              "Skipping PyTorch transforms for the masks.",
                              UserWarning
                              )
            image = torch.from_numpy(image).type(torch.float32)
            image = image.view(1, 1, *image.shape)
            image = self.transform['pytorch'](image)
            image = image.view(*image.shape[-2:])

        return image, mask

    @classmethod
    def read_mammogram(cls, file_path):
        """
        Reads a mammography image from file and returns the pixel array along with DICOM metadata (if available).

        Supported image file formats: .dcm, .dicom, .png, .npy

        Parameters
        ----------
        file_path : str
            Path to the image file.

        Returns
        -------
        tuple
            image : numpy.ndarray
                The pixel array of the image.
            dicom_data : pydicom.dataset.FileDataset or None
                The DICOM metadata, or None if the file is not a DICOM format.

        Raises
        ------
        Exception
            If the file extension is not supported.
        """

        dicom_data = None

        if file_path.endswith('.dcm') or file_path.endswith('.dicom'):
            image, dicom_data = cls.__read_dicom(file_path)
        elif file_path.endswith('.png'):
            image = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
            assert image.dtype == np.uint16, (f'Wrong data type. '
                                              f'Data type of the loaded file is {image.dtype}')
        elif file_path.endswith('.npy'):
            with open(file_path, 'rb') as f:
                image = np.load(f)
        else:
            raise Exception('Dataset class only accept files with the .dcm, .dicom, .png, and .npy extension. '
                            'Check the type and extension of the file: {}'.format(file_path))

        return image, dicom_data

    @staticmethod
    def __read_dicom(path):
        """
        Reads a DICOM file and extracts the image and metadata.

        This method attempts to read a DICOM file using the `pydicom` library.
        If the file is invalid, it uses the `force` parameter to try reading non-compliant files.
        It also handles missing or invalid `TransferSyntaxUID` attributes to ensure the pixel array can be extracted.

        Parameters
        ----------
        path : str
            The file path to the DICOM file.

        Returns
        -------
        tuple
            A tuple containing:
            - image : numpy.ndarray
                The pixel array extracted from the DICOM file.
            - dicom_data : pydicom.dataset.FileDataset
                The DICOM dataset object containing metadata and other information.

        Raises
        ------
        InvalidDicomError
            If the file cannot be read as a valid DICOM, even with `force=True`.
        AttributeError
            If the file lacks the `pixel_array` attribute and cannot be corrected.
        Exception
            If any other unexpected error occurs during file reading or processing.

        Notes
        -----
        - If the `TransferSyntaxUID` is missing, it is set to `ExplicitVRLittleEndian` by default.
        - This method is intended for internal use and is marked as private by convention.

        Examples
        --------
        >>> path = "example.dcm"
        >>> image, dicom_data = ExampleClass.__read_dicom(path)
        >>> print(image.shape)
        (512, 512)
        """

        try:
            dicom_data = pydicom.dcmread(path)
        except InvalidDicomError:
            dicom_data = pydicom.dcmread(path, force=True)
        except Exception as e:
            print(f"An Error occurred: {e}")
            raise

        try:
            image = dicom_data.pixel_array
        except AttributeError as e:
            if "TransferSyntaxUID" in str(e):
                file_meta = FileMetaDataset()
                file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                dicom_data.file_meta = file_meta
                image = dicom_data.pixel_array
            else:
                raise

        return image, dicom_data


class ClassificationDataset(BaseDataset):
    """
    Classification dataset interface for mammography data.

    This class reads the metadata of training samples and enables returning
    `breast_id`, `image`, `label`, and optionally `domain_label`. If transformations
    are provided, they are applied to the returned image.

    Supports integration with PyTorch's Dataset API.
    """

    def __init__(self, metadata_path, transform=None):
        super().__init__(metadata_path, transform)

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset, including the mammography image and its associated metadata.

        Ensures that required metadata columns are present: 'BreastID', 'FilePath', 'OneHotLabel', and
        'ImageLaterality'.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            breast_id : str
                Identifier for the breast/image.
            image : torch.Tensor or numpy.ndarray
                Transformed mammography image.
            label : list or torch.Tensor
                One-hot encoded label for the image.
            domain_label : int, optional
                Optional domain label if available.
        """

        file_path = self.metadata.at[index, 'DICOM']
        image, dicom_data = self.read_mammogram(file_path)

        breast_id = self.metadata.at[index, 'BreastID']
        one_hot_label = torch.Tensor(self.metadata.at[index, 'OneHotLabel'])
        image_laterality = self.metadata.at[index, 'ImageLaterality']
        # view_position = self.metadata.at[index, 'ViewPosition']

        if self.transform:
            image, _ = self._apply_transform(image,
                                             laterality=image_laterality,
        #                                     view_position=view_position,
                                             mask=None
                                             )

        return_values = (breast_id, image, one_hot_label)

        if 'DomainLabel' not in self.metadata.keys():
            return return_values
        else:
            domain_label = torch.Tensor(self.metadata.at[index, 'DomainLabel'])
            return *return_values, domain_label


class SegmentationDataset(BaseDataset):
    """
    Dataset class for mammography image segmentation tasks.

    This class extends the `BaseDataset` to support loading both images and corresponding segmentation masks.
    It reads metadata from an Excel file and applies transformations (if provided) to the images and masks.

    Returns samples as a tuple of breast ID, transformed image, label, and segmentation mask.
    """

    def __init__(self, metadata_path, transform=None):
        super().__init__(metadata_path, transform)

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset including the breast_id, image, label, and segmentation mask.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            breast_id : str
                Identifier for the breast/image.
            image : torch.Tensor or numpy.ndarray
                Transformed mammography image.
            one_hot_label : torch.Tensor
                One-hot encoded label corresponding to the image.
            mask : torch.Tensor or numpy.ndarray
                Transformed segmentation mask associated with the image.
        """

        file_path = self.metadata.at[index, 'DICOM']
        image, dicom_data = self.read_mammogram(file_path)

        mask_path = self.metadata.at[index, 'Mask']
        mask = self._read_mask(mask_path)

        breast_id = self.metadata.at[index, 'BreastID']
        image_laterality = self.metadata.at[index, 'ImageLaterality']
        one_hot_label = torch.Tensor(self.metadata.at[index, 'OneHotLabel'])
        view_position = self.metadata.at[index, 'ViewPosition']

        if self.transform:
            image, mask = self._apply_transform(image,
                                                laterality=image_laterality,
                                                view_position=view_position,
                                                mask=mask)

        return_values = (breast_id, image, one_hot_label, mask)

        return return_values

    @staticmethod
    def _read_mask(mask_path):
        """
        Reads a segmentation mask from the given file path and returns it as a PyTorch tensor.

        Supported formats include `.npy`, `.npz`, DICOM (`.dcm`), and
        common image formats (e.g., `.png`).

        Parameters
        ----------
        mask_path : str
            Path to the mask file.

        Returns
        -------
        mask : torch.Tensor
            Segmentation mask as a tensor with dtype `torch.uint8`.
            If the mask is 2D, it will be expanded to include a channel dimension.
        """

        if mask_path.endswith('npy'):
            mask = np.load(mask_path)
        elif mask_path.endswith('npz'):
            mask = np.load(mask_path)
            mask = mask.f.masks
        elif 'dcm' in mask_path:
            mask = pydicom.dcmread(mask_path).pixel_array
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        mask = torch.Tensor(mask).to(torch.uint8)

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        return mask


class ObjectDetectionDataset(BaseDataset):
    """
    Dataset class for object detection tasks.

    This class extends the `BaseDataset` to support loading images along with their associated bounding boxes and labels.
    It reads metadata from an Excel file and applies transformations (if provided) to the images and annotations.

    Returns samples as a tuple of breast_id, image, one_hot_label, and bounding box coordinates.

    .. warning::
        This class is currently under development and may contain bugs. Use with caution.
    """

    def __init__(self, metadata_path, transform=None):
        super().__init__(metadata_path, transform)

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset, including the image, one-hot label, and bounding boxes.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            breast_id : str
                Unique identifier for the breast/image.
            image : torch.Tensor
                Transformed mammography image.
            one_hot_label : torch.Tensor
                One-hot encoded label corresponding to the image.
            bounding_boxes : torch.Tensor
                Bounding boxes for the object detection task, if available.
        """

        file_path = self.metadata.at[index, 'DICOM']
        image, dicom_data = self.read_mammogram(file_path)

        coordinates = self.metadata.at[index, 'Coordinates']
        breast_id = self.metadata.at[index, 'BreastID']
        one_hot_label = torch.Tensor(self.metadata.at[index, 'OneHotLabel'])

        coordinates = literal_eval(coordinates)
        labels = []
        bboxes = []

        for label, x, y, width, height in coordinates:
            if label == 0:
                labels.append([1., 0.])
            else:
                labels.append([0., 1.])
            bboxes.append((x, y, width, height))

        if self.transform:
            image, bboxes, labels, mask = self._apply_transform(image, bboxes, labels)

        return_values = (breast_id, image, one_hot_label, torch.tensor(bboxes))

        return return_values

    def _apply_transform(self, image, bboxes, labels, mask):
        """
        Apply transformations to the image, bounding boxes, class labels, and mask using the Albumentations library.

        Parameters
        ----------
        image : numpy.ndarray
            The input DICOM image data.
        bboxes : list of list of float
            List of bounding boxes in the format [x_min, y_min, x_max, y_max].
        labels : list of int
            List of class labels corresponding to the bounding boxes.
        mask : numpy.ndarray
            The segmentation mask associated with the image.

        Returns
        -------
        tuple
            A tuple containing:
                - image (numpy.ndarray): The transformed image.
                - bboxes (list of list of float): The transformed bounding boxes.
                - labels (list of int): The transformed class labels.
                - mask (numpy.ndarray): The transformed mask.
        """

        transformed = self.transform(image=image, bboxes=bboxes, labels=labels, masks=mask)
        image = transformed['image']
        bboxes = transformed['bboxes']
        labels = transformed['labels']
        mask = transformed['masks']

        return image, bboxes, labels, mask
