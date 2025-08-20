import warnings

import cv2
import numpy as np


__all__ = ['get_one_hot', 'freeze_layers', 'unfreeze_all', 'detect_breast_laterality']


def get_one_hot(y_pred, nb_classes):
    """Creates one hot encoded matrix from 2D logit matrix.

    Example
    --------
    [[0.95, 0.045, 0.005]       --> [[1, 0, 0]]
     [0.10, 0.12, 0.78]]            [[0, 0, 1]]

    Parameters
    ----------
    y_pred: numpy.array or torch.Tensor
    nb_classes: int

    Returns
    -------
    one_hot: numpy.array
    """

    indices = np.argmax(y_pred, axis=1)
    one_hot = np.eye(nb_classes)[indices]

    return one_hot


def freeze_layers(model, layer_list):
    """
    Freezes the weights of specified layers in the model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model containing layers to be frozen.
    layer_list : tuple or list of torch.nn.Module types
        A tuple or list of layer classes (e.g., (nn.Linear, nn.Conv2d)) to freeze.

    Returns
    -------
    None
        The model is modified in place, with the `requires_grad` attribute set to `False` for matching layers.

    Notes
    -----
    This function recursively iterates through all named submodules in the model.
    If a module matches any type in `layer_list` and has a `weight` attribute, its gradients are disabled.
    """

    try:
        for name, module in model.named_modules():
            if isinstance(module, layer_list):
                if hasattr(module, "weight"):
                    module.weight.requires_grad = False
                    if module.bias is not None:
                        module.bias.requires_grad = False
                else:
                    warnings.warn(
                        f"Layer '{name}' ({module.__class__.__name__}) does not have "
                        f"a 'weight' attribute. Skipping freezing.",
                        UserWarning
                    )
        layer_types = ', '.join([layer.__name__ for layer in layer_list])
        print('Types of model layers that are frozen: {}'.format(layer_types))
    except Exception as e:
        print(e)


def unfreeze_all(model):
    """
    Unfreezes all layers in the model by setting `requires_grad` to `True` for all parameters.

    This function iterates through each child module in the given model and enables gradient computation
    for all its parameters.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model whose parameters will be unfrozen.
    """

    for layer_name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad_(True)


def detect_breast_laterality(grayscale_image):
    """
    Detects the laterality (left or right) of breast tissue in a grayscale image.

    Parameters
    ----------
    grayscale_image: np.ndarray

    Returns
    -------
    str
        'Left' or 'Right'
    """

    grayscale_image = cv2.normalize(grayscale_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_8U)

    # For simplicity, use a basic thresholding approach
    _, binary_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour corresponds to the breast region
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the centroid of the largest contour
    M = cv2.moments(largest_contour)
    centroid_x = int(M['m10'] / M['m00'])

    # Determine the side based on the centroid position
    img_width, img_height = grayscale_image.shape[::-1]
    laterality = "Left" if centroid_x < img_width / 2 else "Right"

    return laterality
