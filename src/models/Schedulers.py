import numpy as np


class LambdaScheduler:
    """
    A class to schedule lambda values for domain adaptation training.

    Attributes
    ----------
    len_dataloader : int
        The length of the dataloader (number of batches per epoch).
    n_total_iteration : int
        The total number of iterations (batches) for the entire training.
    up_limit : float
        The upper limit of the function. `up_limit` takes values between 0 and 1.

    Methods
    -------
    get_alpha(batch_step, curr_epoch):
        Computes the lambda value for the current batch and epoch.
    """

    def __init__(self, len_dataloader, n_total_epoch, t_max=1.):
        """
        Parameters
        ----------
        len_dataloader : int
            The length of the dataloader (number of batches per epoch).
        n_total_epoch : int
            The total number of epochs for training.
        t_max : float
            The upper limit of the function. `t_max` takes values between 0 and 1.
        """

        assert (t_max >= 0.) & (t_max <= 1.), ('`up_limit` must be greater than zero, equal to zero,'
                                                     ' and lower than or equal to 1.')

        self.len_dataloader = len_dataloader
        self.n_total_iteration = self.len_dataloader * n_total_epoch
        self.t_max = t_max

    def get_lambda(self, batch_step, curr_epoch):
        """
        Compute the lambda value for the current batch and epoch.

        Parameters
        ----------
        batch_step : int
            The current batch index within the epoch.
        curr_epoch : int
            The current epoch number (starting from 1).

        Returns
        -------
        float
            The lambda value for the current training iteration.
        """

        p = float(batch_step + (curr_epoch) * self.len_dataloader) / self.n_total_iteration
        lambda_domain = (2 * self.t_max) / (1 + np.exp(-10 * p)) - self.t_max

        return lambda_domain
