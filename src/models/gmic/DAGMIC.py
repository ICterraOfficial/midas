import torch
from torch.autograd import Function
import torch.nn as nn

from .gmic import GMIC


class GradientReversalLayer(Function):
    """
    Implements a Gradient Reversal Layer (GRL) used in domain adversarial training.

    During the forward pass, this layer acts as an identity function.
    During the backward pass, it reverses the gradients by multiplying them with a negative scalar (-lambda_domain),
    enabling the model to learn domain-invariant features.

    Methods
    -------
    forward(ctx, x, lambda_domain):
        Stores lambda_domain in the context and returns the input unchanged.

    backward(ctx, grad_output):
        Multiplies the incoming gradient by -lambda_domain to reverse it and returns the result.
    """

    @staticmethod
    def forward(ctx, x, lambda_domain):
        """
        Forward pass of the Gradient Reversal Layer.

        Parameters
        ----------
        ctx : context object
            Context object for saving information for backward computation.
        x : torch.Tensor
            Input tensor.
        lambda_domain : float
            Coefficient used to scale the reversed gradients.

        Returns
        -------
        torch.Tensor
            Output tensor, same as input.
        """

        ctx.lambda_domain = lambda_domain
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass that reverses the gradient.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        torch.Tensor
            Reversed gradient scaled by -lambda_domain.
        None
            No gradient needed for lambda_domain.
        """

        output = grad_output.neg() * ctx.lambda_domain

        return output, None


class DomainAdversarialGMIC(GMIC):
    """
     A class that extends the GMIC model to incorporate domain adversarial training for domain adaptation tasks.

     This model includes an additional domain-classifier head that allows the network
     to learn domain-invariant features.
     The domain classifier aims to differentiate between domains and the Gradient Reversal Layer (GRL)
     helps reverse gradients during backpropagation, enabling domain-invariant feature learning.

     Parameters
     ----------
     parameters : dict
         Dictionary containing model configuration and hyperparameters, including:
         - 'num_classes' : int
             Number of output classes for the domain classifier.

     Attributes
     ----------
     domain_classifier : nn.Sequential
         The domain classifier network consisting of linear layers and ReLU activations.
     """

    def __init__(self, parameters):
        super(DomainAdversarialGMIC, self).__init__(parameters)

        # domain-classifier branch
        self.domain_classifier = nn.Sequential(
            nn.Linear(768, 256, bias=False),
            nn.ReLU(256),
            nn.Linear(256, 128, bias=False),
            nn.ReLU(128),
            nn.Linear(128, parameters["num_classes"], bias=False),
        )

    def forward(self, x_original, lambda_domain=0.):
        """
                :param x_original: N,H,W,C numpy matrix
                """
        # global network: x_small -> class activation map
        self.h_g, self.saliency_map = self.global_network.forward(x_original)

        # calculate y_global
        # note that y_global is not directly used in inference
        self.y_global = self.aggregation_function.forward(self.saliency_map)

        # region proposal network
        small_x_locations = self.retrieve_roi_crops.forward(x_original, self.cam_size, self.saliency_map)

        # convert crop locations that is on self.cam_size to x_original
        self.patch_locations = self._convert_crop_position(small_x_locations, self.cam_size, x_original)

        # patch retriever
        crops_variable = self._retrieve_crop(x_original, self.patch_locations, self.retrieve_roi_crops.crop_method)
        # self.patches = crops_variable.data.cpu().numpy()

        # detection network
        batch_size, num_crops, I, J = crops_variable.size()
        crops_variable = crops_variable.view(batch_size * num_crops, I, J).unsqueeze(1)
        h_crops = self.local_network.forward(crops_variable).view(batch_size, num_crops, -1)

        # MIL module
        # y_local is not directly used during inference
        z, self.patch_attns, self.y_local = self.attention_module.forward(h_crops)

        # fusion branch
        # use max pooling to collapse the feature map
        g1, _ = torch.max(self.h_g, dim=2)
        global_vec, _ = torch.max(g1, dim=2)
        concat_vec = torch.cat([global_vec, z], dim=1)
        self.y_fusion = torch.sigmoid(self.fusion_dnn(concat_vec))

        grl_output = GradientReversalLayer.apply(concat_vec, lambda_domain)
        self.y_domain = torch.softmax(self.domain_classifier(grl_output), dim=-1)

        outputs = {
            'global': self.y_global,
            'local': self.y_local,
            'fusion': self.y_fusion,
            'saliency_map': self.saliency_map,
            'domain': self.y_domain
        }

        return outputs
