import torch
import tqdm

from .EvaluationTools import MetricCalculator


class Evaluator:
    """
    A class used to evaluate model.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to evaluate.
    data_loader : torch.utils.data.DataLoader
        DataLoader for the evaluation dataset.
    """

    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.batch_size = self.data_loader.batch_size
        self.device = next(self.model.parameters()).device

        # Store predictions per breast_id
        self.metric_calculator = MetricCalculator()

    def __crate_prog_bar(self, prog_bar):
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
        prog_bar.set_description('Validation  ')
        prog_bar.set_postfix_str('            ')

        return prog_bar

    def evaluate(self, eval_module='fusion', prog_bar=True):
        """
        Evaluate the model on the evaluation dataset.

        Parameters
        ----------
        eval_module: str, optional
            Specifies which output module of the GMIC to use for evaluation. The default is 'fusion'.
        prog_bar : bool, optional
            Whether to display the progress bar during evaluation. Default is True.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics including fprs, tprs, roc_auc, precision, recalls, and pr_auc.
        """

        prog_bar = self.__crate_prog_bar(prog_bar)

        self.model.eval()
        with torch.no_grad():
            for i, (breast_ids, batch_image, batch_true) in prog_bar:
                batch_image = batch_image.unsqueeze(1).to(self.device)
                # Forward pass.
                predictions = self.model(batch_image)
                self.metric_calculator.store_preds_truths(breast_ids, predictions[eval_module], batch_true)

        metrics = self.metric_calculator.calculate_metrics(clear_cache=True)

        return metrics


class DDPEvaluator(Evaluator):
    def __init__(self):
        # super(DDPEvaluator, self).__init__(model, data_loader)
        # This class will be implemented for distributed model training on multiple GPUs.
        raise NotImplementedError
