from copy import deepcopy
import torch.nn as nn


class HookHandler:
    """
    A utility class for registering hooks on a neural network model to capture
    activations and gradients of specified layers during forward and backward passes.

    Attributes
    ----------
    activations : dict
        A dictionary to store the output activations of the layers.
    gradients : dict
        A dictionary to store the gradients of the layers.
    """

    def __init__(self, model, layer_type=(nn.Linear, nn.Conv2d)):
        """
        Initializes the HookHandler with the provided model and registers hooks.

        Parameters
        ----------
        model : nn.Module
            The PyTorch model on which to register the hooks.
        """

        self.activations = {}
        self.gradients = {}

        if isinstance(layer_type, type) and issubclass(layer_type, nn.Module):
            # Single nn.Module subclass passed, wrap it in a tuple
            self.layer_type = (layer_type,)
        elif isinstance(layer_type, tuple):
            # Ensure all elements in the tuple are nn.Module subclasses
            if not all(isinstance(t, type) and issubclass(t, nn.Module) for t in layer_type):
                raise TypeError("All elements in layer_type must be nn.Module subclasses.")
            else:
                self.layer_type = layer_type
        else:
            raise TypeError("layer_type should be an nn.Module subclass or a tuple of them.")

        self.register_hooks(model)

    def get_activations(self):
        """
        Retrieves the captured activations from the model layers.

        Returns
        -------
        dict
            A dictionary containing the activations of the registered layers.
        """

        return deepcopy(self.activations)

    def get_gradients(self):
        """
        Retrieves the captured gradients from the model layers.

        Returns
        -------
        dict
            A dictionary containing the gradients of the registered layers.
        """

        return deepcopy(self.gradients)

    def set_activation_hook(self, name):
        """
        Creates a hook function to capture the output activations of a specified layer.

        Parameters
        ----------
        name : str
            The name of the layer whose output is to be captured.

        Returns
        -------
        function
            A hook function that captures the output of the specified layer and stores it
            in the `activations` dictionary under the given name.
        """

        def hook(module, input, output):
            self.activations[name] = output.clone().detach().cpu()

        return hook

    def set_gradient_hook(self, name):
        """
        Creates a hook function to capture the gradients of a specified layer.

        Parameters
        ----------
        name : str
            The name of the layer whose gradients are to be captured.

        Returns
        -------
        function
            A hook function that captures the gradients of the specified layer and stores
            them in the `gradients` dictionary under the given name.
        """

        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output

        return hook

    def register_hooks(self, module, parent_name=''):
        """
        Recursively registers hooks on convolutional and linear layers within the model.

        Parameters
        ----------
        module : nn.Module
            The model or layer whose child modules are to be examined.
        parent_name : str, optional
            The prefix for the layer names (default is '').

        Notes
        -----
        This function traverses the model recursively, registering forward hooks for
        `nn.Conv2d` and `nn.Linear` layers to capture their outputs and backward hooks
        to capture their gradients.

        Examples
        --------
        >>> import torch.nn as nn
        >>> inp_data = torch.rand(1, 1, 32, 32)
        >>> model = nn.Sequential(
        ...     nn.Conv2d(1, 2, 3),
        ...     nn.ReLU(),
        ...     nn.Conv2d(2, 4, 3),
        ...     nn.ReLU()
        ... )
        >>> hook_handler = HookHandler(model)
        >>> out = model(inp_data)
        >>> out.sum().backward()
        >>> activations = hook_handler.get_activations()
        >>> gradients = hook_handler.get_gradients()
        >>> layer_name = '0'
        >>> print(activations[layer_name])
        tensor([[[[]]]])
        >>> print(gradients.keys())
        tensor([[[[]]]])
        """

        for name, param in module.named_children():
            name = name if parent_name == '' else '.'.join([parent_name, name])
            if isinstance(param, self.layer_type):
                param.register_forward_hook(self.set_activation_hook(name))
                param.register_full_backward_hook (self.set_gradient_hook(name))
            else:
                self.register_hooks(param, name)
