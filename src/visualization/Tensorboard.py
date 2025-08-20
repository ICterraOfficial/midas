import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    """
    This class provides methods for logging data to TensorBoard, which is useful for
    visualizing training progress and metrics.
    """

    def __init__(self, log_dir="events"):
        """Initialize summary writer.
        Parameters
        ----------
        log_dir: str
            Storage path of the event files.
        """

        self.step = 0
        self.writer = SummaryWriter(log_dir)

    def close(self):
        """Close summary writer."""

        self.writer.close()

    def add_images(self, tag, image):
        """
        Adds images into tensorboard.

        Parameters
        ----------
        tag: str
            Data identifier.
        image: torch.Tensor
            [x, y, z] shape.
        slices: dict of ints
            Start and end point of the image slice, and step size:
        Returns
        -------
        None
        """

        image = image.unsqueeze(1)
        img_grid = torchvision.utils.make_grid(image)
        self.writer.add_image(tag, img_grid)

    def add_graph(self, model, input_size=(1, 2294, 1914), device='gpu'):
        """
        Creates an input to model and adds model graph into Tensorboard.

        Parameters
        ----------
        model: torch.nn.Module or inherited it.
        input_size: tuple of ints
        device: str
            "cpu" or "cuda"
        Returns
        -------
        None
        """

        inp = torch.rand((1, *input_size), requires_grad=False, dtype=torch.float).to(device)

        self.writer.add_graph(model, inp, use_strict_trace=False)

    def add_lr(self, lr):
        self.writer.add_scalar("Learning rate", lr, self.step)

    def add_metric(self, name, value):
        self.writer.add_scalar(name, value, self.step)

    def add_train_loss(self, module_name, loss):
        self.writer.add_scalar(f"Training loss/{module_name}", loss, self.step)

    def add_val_loss(self, module_name, loss):
        self.writer.add_scalar(f"Validation loss/{module_name}", loss, self.step)

    def add_scalars(self, step, lr=None, metrics=None, train_loss=None,
                    val_loss=None, roc_auc=None, pr_auc=None, data_split=None):
        """
        Add scalars into tensorboard.

        Parameters
        ----------
        step: int
        lr: float
            Learning rate.
        metrics: dict of dicts
            {'category_name':{'accuracy': value,
                    'precision': value,
                    'sensitivity': value,
                    'specificity': value}}
        train_loss: float
        val_loss: float

        Returns
        -------
        None.
        """

        self.step = step

        if lr:
            self.add_lr(lr)

        if metrics:
            for module_name, module_metrics in metrics.items():
                for category, mets in module_metrics.items():
                    for name, value in mets.items():
                        self.add_metric(module_name.capitalize()+' module/'+'Class: '+category+'/'+name, value)

        if isinstance(train_loss, dict):
            for module_name, loss_value in train_loss.items():
                self.add_train_loss(module_name, loss_value)
        elif isinstance(train_loss, (float, torch.Tensor)):
            self.add_metric('Training loss/total_loss', train_loss)

        if val_loss:
            for module_name, loss_value in val_loss.items():
                self.add_val_loss(module_name, loss_value)

        if roc_auc:
            self.add_metric('AUC/{}:ROC_AUC'.format(data_split), roc_auc)

        if pr_auc:
            self.add_metric('AUC/{}:PR_AUC'.format(data_split), pr_auc)

    def add_text(self, tag, text):
        self.writer.add_text(tag, text)

    def flush(self):
        self.writer.flush()