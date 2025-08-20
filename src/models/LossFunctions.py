import numpy as np
import torch


class GMICLoss(torch.nn.Module):
    """Loss function for end-to-end training of GMICLoss model.

    Notes
    -----
    This loss function implemented according to formulation in the paper of GMIC.
    See details: https://doi.org/10.1016/j.media.2020.101908

    Examples
    --------
    import numpy as np
    from models.gmic.LossFunctions import GMICLoss
    y_pred # output of the GMIC model
    y_actual # actual label of the input image
    criterion = GMICLoss()
    module_losses, total_loss = criterion(y_pred, y_actual)
    print(module_losses)
    print(total_loss)
    torch.Tensor(0.75)
    prediction={'global': torch.Tensor shape of (0.44),
                'local': torch.Tensor shape of (0.01),
                'fusion': torch.Tensor shape of (0.2)}
    """

    def __init__(self, beta=np.power(10, -4.5), pos_weight=None):
        """Defines GMIC loss function parameters.

        Parameters
        ----------
        beta : float
            The factor for how much the salience map affects the total loss value.
        pos_weight : int
            The weight of the positive label in the loss function.
        """

        super().__init__()

        self.beta = beta
        self.pos_weight = pos_weight

        if self.pos_weight is None:
            self.BCELoss = torch.nn.BCELoss()
        else:
            self.BCELoss = WeightedBCELoss(pos_weight=pos_weight)

    def _min_max_normalize_image(self, image, min_value=0.0, max_value=1.0):
        """
        Apply min-max normalization separately along the batch size and each channel (vectorized).

        Parameters:
        - image (torch.Tensor): Input image tensor with shape [bs, channel_size, height, width].
        - min_value (float): Minimum value for normalization.
        - max_value (float): Maximum value for normalization.

        Returns:
        - torch.Tensor: Normalized image tensor.
        """

        # Reshape the image tensor to combine batch and channel dimensions
        flattened_image = image.view(image.size(0), image.size(1), -1)

        # Calculate min and max values along the third dimension (height * width)
        min_values, _ = flattened_image.min(dim=2, keepdim=True)
        max_values, _ = flattened_image.max(dim=2, keepdim=True)

        # Apply min-max normalization to the entire flattened image tensor
        normalized_flattened_image = (flattened_image - min_values) / (max_values - min_values) * (
                    max_value - min_value) + min_value

        # Reshape the normalized flattened image back to the original shape
        normalized_image = normalized_flattened_image.view_as(image)

        return normalized_image

    def forward(self, prediction, target):
        """Calculates loss values of the GMIC modules separately and total loss value of GMIC model.

        Parameters
        ----------
        prediction : dict of tensors
            Outputs of the GMIC model.
            prediction={'global': torch.Tensor shape of (1,2),
                        'local': torch.Tensor shape of (1,2),
                        'fusion': torch.Tensor shape of (1,2),
                        'saliency_map': torch.Tensor shape of (1,2,height,width)}
        target: torch.Tensor
            Actual label in one hot format.

        Returns
        -------
        module_losses : dict of torch.Tensors
            prediction={'global': torch.Tensor shape of (1),
                        'local': torch.Tensor shape of (1),
                        'fusion': torch.Tensor shape of (1)}
        total_loss : torch.Tensor
            torch.Tensor shape of (1)
        """

        total_loss = 0
        saliency_maps = prediction.pop('saliency_map')
        module_losses = {'global': None, 'local': None, 'fusion': None}

        for module_name, outputs in prediction.items():
            module_loss = self.BCELoss(outputs, target)
            module_losses[module_name] = module_loss
            total_loss += module_loss

        # normalized_smaps = self._min_max_normalize_image(saliency_maps)
        l1_regularization = self.beta * torch.norm(saliency_maps, 1)

        total_loss += l1_regularization

        return module_losses, total_loss

    def __str__(self):
        key = self.__class__.__name__
        params = 'Beta: {} Module Loss Function: {}, Pos Weight: {}'.format(self.beta,
                                                                            str(self.BCELoss), str(self.pos_weight))
        return '{}: {}'.format(key, params)


class DALoss(GMICLoss):
    """
    Domain-Adversarial Loss class for training with domain adaptation.

    This class extends `GMICLoss` by introducing an additional domain loss component,
    enabling domain-adversarial training. It is designed to help the model generalize better
    across different data distributions by penalizing domain-specific features.

    Parameters
    ----------
    beta : float, optional
        Regularization strength for L2 loss term. Default is 10^-4.5.
    lambda_domain : float, optional
        Scaling factor for the domain adversarial loss. Controls the impact of domain loss
        in the total loss computation. Default is 1.0.
    pos_weight : torch.Tensor or None, optional
        Weight for positive class in binary classification, used to handle class imbalance
        in classification loss computation.
    """

    def __init__(self, beta=np.power(10, -4.5), lambda_domain=1.0, pos_weight=None):
        super(DALoss, self).__init__(beta=beta, pos_weight=pos_weight)

        self.beta = beta
        self.lambda_domain = lambda_domain

    def forward(self, predictions_class, batch_gt_label, predictions_domain, batch_domain_label, saliency_maps):
        total_class_loss = 0

        # Calculate total class loss for global, local, and fusion modules.
        for module_name, module_output in predictions_class.items():
            total_class_loss += self.BCELoss(module_output, batch_gt_label)

        # Calculate domain loss.
        domain_loss = self.BCELoss(predictions_domain, batch_domain_label)

        # Calculate smap loss.
        # To equalize scales of benign and malign classes apply min-max normalization.
        normalized_smaps = self._min_max_normalize_image(saliency_maps)
        smap_loss = self.beta * torch.norm(normalized_smaps, 1)

        # Sum up all losses.
        total_loss = total_class_loss + smap_loss + self.lambda_domain * domain_loss

        loss_values = {'class_loss': total_class_loss,
                       'domain_loss': domain_loss,
                       'total_loss': total_loss}

        return loss_values


class WeightedBCELoss:
    def __init__(self, pos_weight=1.0):
        self.pos_weight = pos_weight

    def __call__(self, predictions, targets):
        """
        Compute the weighted binary cross-entropy loss for sigmoid outputs.

        Parameters
        ----------
            predictions (torch.Tensor): Sigmoid outputs (probabilities) from the model, shape (N,).
            targets (torch.Tensor): Ground truth binary labels, shape (N,).

        Returns
        --------
            torch.Tensor: The scalar loss value.
        """

        assert predictions.shape == targets.shape, \
            f'A target size {targets.shape} does not match prediction size {predictions.shape}.'

        # Clamp predictions to avoid log(0)
        epsilon = 1e-7  # Small value to ensure numerical stability
        predictions = torch.clamp(predictions, epsilon, 1 - epsilon)

        # Compute weighted binary cross-entropy
        loss = -(targets * self.pos_weight * torch.log(predictions) +
                 (1 - targets) * torch.log(1 - predictions))

        return loss.mean()

    def __str__(self):
        return 'WeightedBCELoss: {}'.format(self.pos_weight)