from abc import ABC, abstractmethod

import numpy as np
import timm.scheduler.scheduler as timm_schedulers
import torch
import torch.optim.lr_scheduler as torch_schedulers
import tqdm

from .EvaluationTools import MetricCalculator
from .Schedulers import LambdaScheduler


class BaseTrainer(ABC):
    """
    Abstract base class for training deep learning models.

    This class defines the common interface and basic functionality for training workflows,
    including model, optimizer, scheduler, and data loader management. Concrete subclasses
    should implement the `_train` and `fit` methods.

    Parameters
    ----------
    criterion : callable
        The loss function used for training.
    model : torch.nn.Module
        The neural network model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer used for updating model parameters.
    total_epochs : int
        The total number of training epochs.
    data_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler (default is None).
    parallel_mode : bool, optional
        Flag to indicate if model training uses DataParallel or DistributedDataParallel (default is False).
    """

    @abstractmethod
    def __init__(self, criterion, model, optimizer, total_epochs, data_loader, scheduler=None, parallel_mode=False):
        self.curr_epoch = 0
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.total_epochs = total_epochs
        self.scheduler = scheduler
        self.parallel_mode = parallel_mode
        self.device = next(self.model.parameters()).device

    def _crate_prog_bar(self, prog_bar):
        """
        Create and configure the progress bar for training.

        Parameters
        ----------
        prog_bar : bool
            Whether to display the progress bar during training.

        Returns
        -------
        tqdm.tqdm
            Configured progress bar object.
        """

        prog_bar = tqdm.tqdm(enumerate(self.data_loader), total=len(self.data_loader), disable=not(prog_bar))
        # Epoch visualization starts from 1.
        prog_bar.set_description(f"Epoch [{self.curr_epoch+1}/{self.total_epochs}]")
        prog_bar.set_postfix_str(f'Loss: {0}')

        return prog_bar

    def _step_scheduler(self):
        """
        Step the learning rate scheduler based on its type.

        This method handles both PyTorch and timm schedulers. If the scheduler is from
        PyTorch (`torch.optim.lr_scheduler`), it calls `step()` directly. If the scheduler
        is from timm (`timm.scheduler.scheduler`), it calls `step()` with the current epoch.

        Raises
        ------
        ValueError
            If the scheduler is not recognized as either a PyTorch or timm scheduler.

        Notes
        -----
        - `torch_schedulers._LRScheduler` is the base class for PyTorch learning rate schedulers.
        - `timm_schedulers.Scheduler` is the base class for timm learning rate schedulers.
        """

        if isinstance(self.scheduler, torch_schedulers._LRScheduler):
            self.scheduler.step()
        elif isinstance(self.scheduler, timm_schedulers.Scheduler):
            # timm scheduler yielded wrong results when started from 0 at the beginning of trainings.
            # For this reason, it started from 1.
            self.scheduler.step(self.curr_epoch+1)
        else:
            raise ValueError('Scheduler type not supported')

    @abstractmethod
    def _train(self, batch_images, batch_true):
        """
        Perform a single training step.

        Parameters
        ----------
        batch_images : torch.Tensor
            Input images for the current batch.
        batch_true : torch.Tensor
            Ground truth labels for the current batch.

        Returns
        -------
        float
            The total loss for the current batch.
        """

        raise NotImplementedError

    @abstractmethod
    def fit(self, prog_bar=True):
        """
        Train the model for one epoch.

        Parameters
        ----------
        prog_bar : bool, optional
            Whether to display the progress bar during training (default is True).

        Returns
        -------
        dict or None
            A dictionary of metrics like ROC-AUC, PR-AUC, and loss.
        """

        prog_bar = self._crate_prog_bar(prog_bar)
        self.model.train()

        for i, (batch_images, batch_true) in prog_bar:
            loss = self._train(batch_images, batch_true)
            pass

        raise NotImplementedError

    def save_model(self, path):
        """
        Save the trained model to a file.

        Parameters
        ----------
        path : str
            The file path where the model should be saved.
        """

        if self.parallel_mode:
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)

    def __str__(self):
        """
        Returns a string representation of the BaseTrainer class configuration.

        The string provides detailed information about the model, optimizer,
        criterion, total epochs, scheduler (if any), parallel mode status,
        and the device the model is running on.

        Returns
        -------
        str
            A formatted string describing the trainer's configuration, including
            the model, optimizer, criterion, total epochs, scheduler,
            parallel mode, and the device being used.
        """

        trainer_config = (f"BaseTrainer("
                          f"  Model: {self.model.__class__.__name__},"
                          f"  Optimizer: {self.optimizer.__class__.__name__},"
                          f"  Loss Function: {self.criterion.__class__.__name__},"
                          f"  Total Epochs: {self.total_epochs},"
                          f"  Scheduler: {self.scheduler.__class__.__name__ if self.scheduler else 'None'},"
                          f"  Parallel Mode: {self.parallel_mode},"
                          f"  Device: {self.device}"
                          f")")

        return trainer_config


class Trainer(BaseTrainer):
    """
    Trainer class for training a deep learning model.

    This class handles the training loop, validation of training metrics, checkpointing, and learning rate scheduling
    for a given model and dataset

    Parameters
    ----------
    criterion : callable
        The loss function used for training.
    model : torch.nn.Module
        The neural network model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer used for updating model parameters.
    total_epochs : int
        The total number of epochs for training.
    data_loader : torch.utils.data.DataLoader
        The data loader for the training dataset.
    scheduler : torch.optim.lr_scheduler, optional
        The learning rate scheduler (default is None).
    parallel_mode : bool, optional
        Flag to indicate if the model is trained in parallel mode (default is False).
    """

    def __init__(self, criterion, model, optimizer, total_epochs, data_loader, scheduler=None, parallel_mode=False):
        super(Trainer, self).__init__(criterion=criterion, model=model,
                                      optimizer=optimizer, total_epochs=total_epochs,
                                      data_loader=data_loader, scheduler=scheduler,
                                      parallel_mode=parallel_mode)

        self.metric_calculator = MetricCalculator()

    def _train(self, batch_images, batch_true):
        """
        Perform a single training step.

        Parameters
        ----------
        batch_images : torch.Tensor
            Input images for the current batch.
        batch_true : torch.Tensor
            Ground truth labels for the current batch.

        Returns
        -------
        float
            The total loss for the current batch.
        """

        batch_images = batch_images.unsqueeze(1).to(self.device)
        batch_true = batch_true.to(self.device)

        # Forward and backward pass.
        predictions = self.model(batch_images.float())
        module_losses, total_loss = self.criterion(predictions, batch_true)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return predictions, total_loss.item()

    def fit(self, eval_module='fusion', prog_bar=True):
        """
        Train the model for one epoch.

        Parameters
        ----------
        eval_module: str, optional
            Specifies which output module of the GMIC to use for evaluation. The default is 'fusion'.
        prog_bar : bool, optional
            Whether to display the progress bar during training (default is True).

        Returns
        -------
        dict or None
            A dictionary of metrics including ROC-AUC, PR-AUC, and loss if training was successful.
        """

        losses = []
        prog_bar = self._crate_prog_bar(prog_bar)

        self.model.train()
        for i, (breast_ids, batch_images, batch_true) in prog_bar:
            predictions, total_loss = self._train(batch_images, batch_true)
            self.metric_calculator.store_preds_truths(breast_ids, predictions[eval_module], batch_true)
            losses.append(total_loss)
            prog_bar.set_postfix_str(f'Loss: {sum(losses) / len(losses):.4f}')

        if self.scheduler:
            self._step_scheduler()

        metrics = self.metric_calculator.calculate_metrics(clear_cache=True)
        metrics.update({'total_loss': sum(losses) / len(losses)})

        return metrics


class DDPTrainer(Trainer):
    def __init__(self):
        # super(DDPTrainer, self).__init__(criterion, model, optimizer, total_epochs, data_loader)
        # This class will be implemented for distributed model training on multiple GPUs.
        raise NotImplementedError


class DomainAdversarialTrainer(BaseTrainer):
    """
    Trainer class for domain adversarial training of deep learning models.

    This class extends the BaseTrainer to support domain adaptation tasks by training the model 
    to minimize classification loss while maximizing domain confusion, encouraging domain-invariant features.
    """

    def __init__(self, criterion, model, optimizer, total_epochs, data_loader, scheduler=None):
        super(DomainAdversarialTrainer, self).__init__(criterion, model, optimizer,
                                                       total_epochs, data_loader, scheduler)

        self.metric_calculator = MetricCalculator()
        self.lambda_scheduler = LambdaScheduler(len_dataloader=len(self.data_loader),
                                                n_total_epoch=self.total_epochs)
    def __strip_target_domain(self, breast_ids, predictions_class, batch_gt_label, batch_domain_label):
        """
        Strip target domain predictions in class predictions and class labels,
        return only source domain class predictions and class labels.

        Parameters
        ----------
        breast_ids: list of str
        predictions_class: dict of torch.Tensor
            Class prediction batch that contains predictions for source and target domain for GMIC modules.
        batch_gt_label: torch.Tensor
            Ground truth labels for source and domain predictions.
        batch_domain_label: torch.Tensor
            Ground truth domain labels for source and target domains.
        Returns
        -------
        predictions_class: torch.Tensor
            Class predictions for only source domain.
        batch_gt_label: torch.Tensor
            Ground truth labels for only source domain.
        """

        # Domain labels are used in BCE Loss, and thus they are one-hot encoded.
        # [1, 0] -> source class, [0, 1] -> target class.
        batch_gt_label = batch_gt_label[batch_domain_label[:, 0] == 1]
        breast_ids = breast_ids[batch_domain_label[:, 0].cpu().numpy() == 1]

        for module_name, module_output in predictions_class.items():
            predictions_class[module_name] = module_output[batch_domain_label[:, 0] == 1]

        stripped_data = {'breast_ids': breast_ids,
                         'predictions_class': predictions_class,
                         'batch_gt_label': batch_gt_label}

        return stripped_data

    def _train(self, breast_ids, batch_images, batch_gt_label, batch_domain_label, lambda_domain):
        """
        Perform a single training step.

        Parameters
        ----------
        breast_ids: list of str
        batch_images: torch.Tensor
            Input images for the current batch.
        batch_gt_label: torch.Tensor
            Ground truth labels for the current batch.
        batch_domain_label: torch.Tensor
            Ground truth domain labels for source and target domains.
        lambda_domain: float
            Lambda value for Gradient Reversal Layer.

        Returns
        -------

        """

        batch_images = batch_images.unsqueeze(1).to(self.device)
        batch_gt_label = batch_gt_label.to(self.device)
        batch_domain_label = batch_domain_label.to(self.device)

        # Forward and backward pass.
        predictions = self.model(batch_images.float(), lambda_domain)
        # First, pop saliency_map
        saliency_maps = predictions.pop('saliency_map')
        # Second, pop domain predictions
        predictions_domain = predictions.pop('domain')
        # Lastly, strip target domain predictions in class predictions and
        # get only source domain predictions for unsupervised training
        stripped_data = self.__strip_target_domain(breast_ids, predictions, batch_gt_label, batch_domain_label)
        predictions_class, batch_gt_label = stripped_data['predictions_class'], stripped_data['batch_gt_label']

        loss_values = self.criterion(predictions_class, batch_gt_label,
                                     predictions_domain, batch_domain_label, saliency_maps)
        self.optimizer.zero_grad()
        loss_values['total_loss'].backward()
        self.optimizer.step()

        return stripped_data, loss_values

    def fit(self, eval_module='fusion', prog_bar=True):
        class_loss_values = []
        domain_loss_values = []
        total_loss_values = []

        prog_bar = self._crate_prog_bar(prog_bar)

        self.model.train()
        for i, (breast_ids, batch_images, batch_gt_label, batch_domain_label) in prog_bar:
            breast_ids = np.array(breast_ids)
            lambda_domain = self.lambda_scheduler.get_lambda(batch_step=i, curr_epoch=self.curr_epoch)
            predictions_source, loss_values = self._train(breast_ids, batch_images, batch_gt_label,
                                                   batch_domain_label, lambda_domain)
            class_loss_values.append(loss_values['class_loss'].detach().cpu().tolist())
            domain_loss_values.append(loss_values['domain_loss'].detach().cpu().tolist())
            total_loss_values.append(loss_values['total_loss'].detach().cpu().tolist())

            self.metric_calculator.store_preds_truths(predictions_source['breast_ids'],
                                                      predictions_source['predictions_class'][eval_module],
                                                      predictions_source['batch_gt_label'])

            prog_bar.set_postfix_str(f'Loss: {sum(total_loss_values) / len(total_loss_values):.4f}')

        if self.scheduler:
            self._step_scheduler()

        metrics = self.metric_calculator.calculate_metrics(clear_cache=True)

        # Only lambda domain value of last iteration.
        metrics.update({'lambda_domain': lambda_domain,
                        'class_loss': sum(class_loss_values) / len(class_loss_values),
                        'domain_loss': sum(domain_loss_values) / len(domain_loss_values),
                        'total_loss': sum(total_loss_values) / len(total_loss_values)})

        return metrics
