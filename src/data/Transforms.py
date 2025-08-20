import random

import cv2
import numpy as np
import torchvision.transforms as t


class Transformer:
    """
    Wrapper class for applying custom transformations to mammography images.

    This class supports applying user-defined transformations that are tailored
    for mammography data. It handles both image and optional mask transformations.

    Parameters
    ----------
    transforms : dict
        Dictionary containing transformation pipelines.
        Expected to have a key 'dicom' with a list of callable transformations:
        {'dicom': [transform1, transform2, ...]}
    """
    def __init__(self, transforms):
        self.transforms = None

        if 'dicom' in transforms.keys() and transforms['dicom'] is not None:
            self.transforms = transforms['dicom']

    def __apply_transforms(self, image, mask=None, **kwargs):
        """
        Apply transformations to the image and optionally to the mask.

        Parameters
        ----------
        image : numpy.ndarray
            Input mammography image.
        mask : numpy.ndarray, optional
            Segmentation mask, if available.
        **kwargs : dict
        Additional keyword arguments to pass to each transform.

        Returns
        -------
        image : numpy.ndarray
            Transformed image.
        mask : numpy.ndarray, optional
            Transformed mask (if provided).
        """

        for t in self.transforms:
            image, mask = t(image, mask=mask, **kwargs)
        if mask is not None:
            return image, mask
        else:
            return image

    def apply(self, image, mask=None, **kwargs):
        """
        Public method to apply the stored transformations.

        Parameters
        ----------
        image : numpy.ndarray
            Input mammography image.
        mask : numpy.ndarray, optional
            Segmentation mask, if available.
        **kwargs : dict
            Additional keyword arguments to pass to each transform.

        Returns
        -------
        image : numpy.ndarray
            Transformed image.
        mask : numpy.ndarray, optional
            Transformed mask (if provided).
        """

        if self.transforms is not None:
            if mask is not None:
                image, mask = self.__apply_transforms(image, mask=mask, coors=None, **kwargs)
            else:
                image = self.__apply_transforms(image, mask=None, coors=None, **kwargs)
        if mask is not None:
            return image, mask
        else:
            return image

class Resize:
    """Resize image according to height and width. Warning: This can distort image!

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Methods
    -------
    __call__(self, image, **kwargs)
        Applies the Resize algorithm to the input image.

    Notes
    -----
    None

    Examples
    --------
    import numpy as np
    from data import Transforms
    image # mammography image
    resizer = Transforms.Resize()
    resized_image = resizer(image)
    """

    def __init__(self, height, width):
        """Images will be resized according to height and width.

        Parameters
        ----------
        height : int
        width : int
        """

        self.height = height
        self.width = width
        self.resizer = t.Resize(size=(height, width))

    def __call__(self, image, **kwargs):
        """Applies the Resize algorithm to the input image.

        Parameters
        ----------
        image : numpy.ndarray
            The input image as a NumPy array. (height, width) shape.

        Returns
        -------
        image : numpy.ndarray
            The input image as a NumPy array. (height, width) shape.
        """

        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if kwargs['mask'] is not None:
            kwargs['mask'] = self.resizer(kwargs['mask'])

        return image, kwargs['mask']

    def __str__(self):
        key = self.__class__.__name__
        params = 'Height: {}, Width: {}'.format(self.height, self.width)
        return '{}: {}'.format(key, params)


class FlipToLeft:
    """Flip the right breast images to left. Breast appears on the left side of the image.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Methods
    -------
    __call__(self, image, **kwargs)
        Applies the FlipToLeft algorithm to the input image.

    Notes
    -----
    None

    Examples
    --------
    import numpy as np
    from data import Transforms
    image # mammography image
    flipper = Transforms.FlipToLeft()
    flipped_image = flipper(image)
    """

    def __init__(self):
        self.flipper = t.RandomHorizontalFlip(p=1)

    def __call__(self, image, **kwargs):
        """Applies the FlipToLeft algorithm to the input image.

        Parameters
        ----------
        image : numpy.ndarray
            The input image as a NumPy array. (height, width) shape.
        kwargs: dict with key 'left'
            If the breast on the left side of the image the 'left' key is True, otherwise False.

        Returns
        -------
        image : numpy.ndarray
            The input image as a NumPy array. (height, width) shape.
        """

        if kwargs['laterality'] == 'L':
            is_left = True
        elif kwargs['laterality'] == 'R':
            is_left = False
        else:
            raise ValueError('laterality must be either L or R')

        if not is_left:
            image = cv2.flip(image, 1)
            if kwargs['mask'] is not None:
                kwargs['mask'] = self.flipper(kwargs['mask'])

        return image, kwargs['mask']

    def __str__(self):
        return self.__class__.__name__


class CropBreastRegion:
    """Detects the breast region in a mammography image and crops the breast area.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Methods
    -------
    __call__(self, image, mask=None, **kwargs)
        Applies the breast region detection and cropping algorithm to the input image.

    Notes
    -----
    This class assumes that the breast is on the left side of the input image. If the breast is on the right side, the
    algorithm may not work properly.

    Examples
    --------
    import numpy as np
    from data import Transforms
    image # mammography image
    cropper = Transforms.CropBreastRegion()
    cropped_image = cropper(image)
    cropped_image.shape
    (1024, 768)
    """

    def __call__(self, image, **kwargs):
        """Applies the breast region detection and cropping algorithm to the input image.

        Parameters
        ----------
        image : numpy.ndarray
            The input image as a NumPy array. (height, width) shape.

        mask : numpy.ndarray or None, optional (default=None)
            The input mask as a NumPy array. If provided, the mask will also be cropped and returned alongside the
            cropped image.

        Returns
        -------
        cropped_image : numpy.ndarray
            The cropped image as a NumPy array.

        cropped_mask : numpy.ndarray or None
            The cropped mask as a NumPy array. Only returned if a mask was provided as input.

        (x, y, w, h) : tuple of ints
            The bounding box of the breast region as a tuple of integers (x, y, w, h), where (x, y) are the coordinates
            of the top-left corner of the bounding box, and (w, h) are the width and height of the bounding box,
            respectively. This tuple can be used to map the coordinates of the cropped image back to the original image.

        Raises
        ------
        None

        """

        # Scale the pixel values to 0-255 range
        img = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply a Gaussian blur to smooth out the image
        blur = cv2.GaussianBlur(img, (5, 5), 0)

        # Apply an adaptive threshold to create a binary image
        # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, thresh = cv2.threshold(blur, 0, 65535, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Find the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Extract the breast region from the original image
        cropped_image = image[y:y + h, x:x + w]

        if kwargs['mask'] is None:
            if kwargs['coors']:
                return cropped_image, kwargs['mask'], (x, y, w, h)
            else:
                return cropped_image, kwargs['mask']
        else:
            cropped_mask = kwargs['mask'][:, y:y + h, x:x + w]
            if kwargs['coors']:
                return cropped_image, cropped_mask, (x, y, w, h)
            else:
                return cropped_image, cropped_mask

    def __str__(self):
        return self.__class__.__name__


class CropBreastRegion2:
    """Detects the breast region in a mammography image and crops the breast area.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Methods
    -------
    __call__(self, image, mask=None, **kwargs)
        Applies the breast region detection and cropping algorithm to the input image.

    Notes
    -----
    This class assumes that the breast is on the left side of the input image. If the breast is on the right side, the
    algorithm may not work properly.

    Examples
    --------
    import numpy as np
    from data import Transforms
    image # mammography image
    cropper = Transforms.CropBreastRegion()
    cropped_image = cropper(image)
    cropped_image.shape
    (1024, 768)
    """

    def __call__(self, image, **kwargs):
        """Applies the breast region detection and cropping algorithm to the input image.

        Parameters
        ----------
        image : numpy.ndarray
            The input image as a NumPy array. (height, width) shape.

        mask : numpy.ndarray or None, optional (default=None)
            The input mask as a NumPy array. If provided, the mask will also be cropped and returned alongside the
            cropped image.

        Returns
        -------
        cropped_image : numpy.ndarray
            The cropped image as a NumPy array.

        cropped_mask : numpy.ndarray or None
            The cropped mask as a NumPy array. Only returned if a mask was provided as input.

        (x, y, w, h) : tuple of ints
            The bounding box of the breast region as a tuple of integers (x, y, w, h), where (x, y) are the coordinates
            of the top-left corner of the bounding box, and (w, h) are the width and height of the bounding box,
            respectively. This tuple can be used to map the coordinates of the cropped image back to the original image.

        Raises
        ------
        None

        """

        # Scale the pixel values to 0-255 range
        img = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply a Gaussian blur to smooth out the image
        blur = cv2.GaussianBlur(img, (5, 5), 0)

        # Apply an adaptive threshold to create a binary image
        # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, thresh = cv2.threshold(blur, 20, 65535, cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, hierarchy = cv2.findContours(255 - thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Find the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Extract the breast region from the original image
        cropped_image = image[y:y + h, x:x + w]

        if kwargs['mask'] is None:
            if kwargs['coors']:
                return cropped_image, kwargs['mask'], (x, y, w, h)
            else:
                return cropped_image, kwargs['mask']
        else:
            cropped_mask = kwargs['mask'][:, y:y + h, x:x + w]
            if kwargs['coors']:
                return cropped_image, cropped_mask, (x, y, w, h)
            else:
                return cropped_image, cropped_mask

    def __str__(self):
        return self.__class__.__name__


class UIntToFloat32:
    """Converts UInt8 type numpy array to Float32 type numpy array.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Methods
    -------
    __call__(self, image, **kwargs)
        Applies the UIntToFloat32 algorithm to the input image.

    Notes
    -----
    None

    Examples
    --------
    import numpy as np
    from data import Transforms
    image # mammography image
    int_to_float = Transforms.UIntToFloat32()
    float_image = int_to_float(image)
    """

    def __call__(self, image, **kwargs):
        """
        Parameters
        ----------
        image : numpy.ndarray
            The input image as a NumPy array. (height, width) shape.

        Returns
        -------
        numpy.float32

        Raises
        -------
        None
        """

        return image.astype(np.float32), kwargs['mask']

    def __str__(self):
        return self.__class__.__name__


class MinMaxNormalization:
    """Normalize image pixels values between 0 and 1.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Methods
    -------
    __call__(self, image, **kwargs)
        Applies the MinMaxNormalization algorithm to the input image.

    Notes
    -----
    None

    Examples
    --------
    import numpy as np
    from data import Transforms
    image # mammography image
    normalizer = Transforms.MinMaxNormalization()
    normalized_image = normalizer(image)
    """

    def __call__(self, image, **kwargs):
        """Applies the MinMaxNormalization algorithm to the input image.

        Parameters
        ----------
        image : numpy.ndarray
            The input image as a NumPy array. (height, width) shape.

        Returns
        -------
        normalized_image : numpy.ndarray

        Raises
        -------
        None
        """

        normalized_image = (image - image.min()) / (image.max() - image.min())
        return normalized_image, kwargs['mask']

    def __str__(self):
        return self.__class__.__name__


class StandardScoreNormalization:
    """Normalize image pixels to have zero mean and unit standard deviation.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Methods
    -------
    __call__(self, dicom_image, **kwargs)
        Applies the StandardScoreNormalization algorithm to the input image.

    Notes
    -----
    This transformation subtracts the mean and divides by the standard deviation
    of the image pixel values.

    Examples
    --------
    import numpy as np
    from data import Transforms
    image # mammography image
    normalizer = Transforms.StandardScoreNormalization()
    normalized_image = normalizer(image)
    """

    def __call__(self, image, **kwargs):
        """Applies the StandardScoreNormalization algorithm to the input image.

        Parameters
        ----------
        image : numpy.ndarray
            The input image as a NumPy array. (height, width) shape.

        Returns
        -------
        normalized_dicom_image : numpy.ndarray

        Raises
        -------
        None
        """

        mean = image.mean()
        std = image.std()

        normalized_image = (image - mean) / std

        return normalized_image, kwargs['mask']

    def __str__(self):
        return self.__class__.__name__


class RandomGaussianNoise:
    """Adds Gaussian noise to given image.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Methods
    -------
    __call__(self, image, **kwargs)
        Applies the Gaussian noise algorithm to the input image.

    Notes
    -----
    None

    Examples
    --------
    import numpy as np
    from data import Transforms
    image # mammography image
    gaussian_noise = Transforms.GaussianNoise()
    noisy_image = gaussian_noise(image)
    """

    def __init__(self, mean=0., std=1.):
        """
        Parameters
        ----------
        mean : float
        std : float
        """

        self.std = std
        self.mean = mean

    def __call__(self, image, **kwargs):
        """Applies the Gaussian noise algorithm to the input image.

        Parameters
        ----------
        image : numpy.ndarray
            (height, width) or (channel, height, width)

        Returns
        -------
        noisy_image : numpy.ndarray

        Raises
        -------
        None
        """
        if random.random() < 0.5:
            noise = np.random.normal(self.mean, self.std, image.shape)
            noisy_image = image + noise

            return noisy_image, kwargs['mask']
        else:
            return image, kwargs['mask']

    def __str__(self):
        key = self.__class__.__name__
        params = 'Mean: {}, Std: {}'.format(self.mean, self.std)
        return '{}: {}'.format(key, params)


class ZeroBackground:
    def __call__(self, image, padding=None, **kwargs):
        """Applies the breast region detection algorithm and setting the background pixels of input image to zero.

        Parameters
        ----------
        image : numpy.ndarray
            The input image as a NumPy array. (height, width) shape.
        padding: int, optional (default=None)
            Amount of padding added around the detected breast region to expand its area.
        mask : numpy.ndarray or None, optional (default=None)
            The input mask as a NumPy array. If provided, the mask will also be cropped and returned alongside the
            cropped image.

        Returns
        -------
        image : numpy.ndarray
            The background fixed image as a NumPy array.

        mask : numpy.ndarray or None
            No change in mask. Only returned if a mask was provided as input.

        Raises
        ------
        None

        """

        # Scale the pixel values to 0-255 range
        img = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply a Gaussian blur to smooth out the image
        blur = cv2.GaussianBlur(img, (105, 105), 0)

        # Apply an adaptive threshold to create a binary image
        # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, thresh = cv2.threshold(blur, 0, 65535, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

        # Define the padding size in pixels
        if padding is not None:

            # Create a kernel for dilation (expands the mask)
            kernel = np.ones((padding, padding), np.uint8)

            # Dilate the mask to add padding
            thresh = cv2.dilate(thresh, kernel, iterations=1)

        # Set background pixels to 0
        mask = thresh.astype(bool)
        image[~mask] = 0

        return image, kwargs['mask']


class ZeroBackground2:
    def __call__(self, image, **kwargs):
        """Applies the breast region detection algorithm and setting the background pixels of input image to zero.

        Parameters
        ----------
        image : numpy.ndarray
            The input image as a NumPy array. (height, width) shape.

        mask : numpy.ndarray or None, optional (default=None)
            The input mask as a NumPy array. If provided, the mask will also be cropped and returned alongside the
            cropped image.

        Returns
        -------
        image : numpy.ndarray
            The background fixed image as a NumPy array.

        mask : numpy.ndarray or None
            No change in mask. Only returned if a mask was provided as input.

        Raises
        ------
        None

        """

        # Scale the pixel values to 0-255 range
        img = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply a median blur to smooth out the image
        blur = cv2.medianBlur(img, 5)

        # Apply an adaptive threshold to create a binary image
        _, thresh = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)

        # Apply opening operation to smooth breast borders
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)

        # Set background pixels to 0
        mask = opening.astype(bool)
        image[~mask] = 0

        return image, kwargs['mask']


class ClipMinMax:
    """
       A transform to preprocess DICOM images by clipping their intensity values
       based on specified percentiles and thresholds.

       Parameters
       ----------
       thresh : int, optional
           Maximum allowable intensity value; images exceeding this threshold
           will undergo clipping (default is 5000).
       first_percentile : float, optional
           Percentile value for lower intensity clipping (default is 0.1).
       second_percentile : float, optional
           Percentile value for upper intensity clipping (default is 99.9).
       """

    def __init__(self, thresh=5000, first_percentile=0.1, second_percentile=99.9):
        self.thresh = thresh
        self.first_percentile = first_percentile
        self.second_percentile = second_percentile

    def __call__(self, image, **kwargs):

        # If the maximum pixel intensity value of image is higher than threshold value apply this transform.
        if image.max() > self.thresh:
            cast = False
            raw_dtype = image.dtype

            # In order to apply clipping properly change data type to int32.
            # Handling subtraction operation on unsigned values is less straightforward than the operation below.
            if image.dtype != np.int32:
                image = image.astype(np.int32)
                cast = True

            # Clip pixel intensities below first percentile ratio to 0.
            clip_low = np.percentile(image[image != 0], self.first_percentile)
            image -= clip_low.astype(np.int32)
            image = np.clip(image, 0, image.max())

            # Clip all the pixel intensities above second percentile ratio to max second percentile value .
            clip_high = np.percentile(image[image != 0], self.second_percentile)
            image[image > clip_high] = clip_high.astype(np.int32)

            # If image data type was different from int32 recover it to original data type.
            if cast:
                image = image.astype(raw_dtype)

        return image, kwargs['mask']

    def __str__(self):
        """
        Return a string representation of the class instance.

        Returns
        -------
        str
            A string describing the instance with its initialization parameters.
        """
        return (
            f"ClipMinMax(thresh={self.thresh}, "
            f"first_percentile={self.first_percentile}, "
            f"second_percentile={self.second_percentile})"
        )
