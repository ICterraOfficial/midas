import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display_html


__all__ = ['plot_grayscale_image', 'plot_roc_curve', 'plot_pr_curve', 'plot_roc_pr',
           'plot_sidebyside', 'plot_histograms_with_cdf']


def plot_grayscale_image(images, figsize=(3.5, 5)):
    """
    Plot a list of images using grayscale colormaps, arranging them in rows and columns.

    Parameters:
    ----------
        images: list, tuple or ndarray
            A list or tuple of NumPy image arrays. Each image array should be 2-dimensional.
        figsize: tuple, optional
            The figure size (width, height) in inches. Default is (3.5, 5).

    Returns
    -------
        None

    This function takes a list of images and arranges them in rows and columns for visualization.
    The maximum number of images displayed per row is 4. If there are more than 4 images,
    new rows are created as needed, each containing up to 4 images.

    The figure size is adjusted based on the number of rows and columns to ensure reasonable image sizes.
    Images are displayed using grayscale colormaps.

    Examples
    --------
    >>>    image_list = (
    >>>        np.random.random((100, 100)),
    >>>        np.random.random((100, 100)),
    >>>        np.random.random((100, 100)),
    >>>        np.random.random((100, 100))
    >>>    )
    >>>    plot_grayscale_image(image_list)
    """

    if not isinstance(images, (list, tuple)):
        images = [images]  # Convert a single image to a list

    num_images = len(images)
    rows = (num_images + 3) // 4  # Calculate the number of rows needed
    cols = min(4, num_images)  # Maximum of 4 columns

    figsize = (cols * figsize[0], rows * figsize[1])  # Adjust figure size based on rows and columns

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = np.array(axs).flatten()  # Convert the axs list to a numpy array and flatten it

    for i, image in enumerate(images):
        axs[i].imshow(image, cmap='gray')
        # axs[i].axis('off')

    # Remove any empty subplots
    for i in range(num_images, rows * cols):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()


def plot_roc_curve(tpr, fpr, roc_auc, lower, upper):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
        tpr: list, np.ndarray
            True Positive Rate values.
        fpr: list, np.ndarray
            False Positive Rate values.
        roc_auc: float
            Area Under the Curve (AUC) value.

    Returns
    -------
        None

    """

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve | AUC {:.2f} 95% CI ({:.2f}, {:.2f})'.format(roc_auc, lower, upper))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_pr_curve(precision, recall, pr_auc):
    """
    Plot the Precision-Recall Curve.

    Parameters
    ----------
        precision: list, np.ndarray
            Precision values.
        recall: list, np.ndarray
            Recall values.
        pr_auc: float
            Area Under the Curve (AUC) value of Precision-Recall.

    Returns
    -------
        None

    """

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label='PR curve | AUC {:.2f})'.format(pr_auc))
    plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower right")
    plt.show()


def plot_roc_pr(roc, pr, title=None, figsize=(16, 7)):
    """
    Plot Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.

    Parameters
    ----------
    roc : dict or tuple of dicts
        Dictionary containing ROC curve information with the following keys:
        - 'fpr' (array-like): False Positive Rate values.
        - 'tpr' (array-like): True Positive Rate values.
        - 'auc' (float): Area under the ROC curve.
        - 'lower' (float): Lower bound of the 95% confidence interval for AUC.
        - 'upper' (float): Upper bound of the 95% confidence interval for AUC.
    pr : dict or tuple of dicts
        Dictionary containing Precision-Recall curve information with the following keys:
        - 'recall' (array-like): Recall values.
        - 'precision' (array-like): Precision values.
        - 'auc' (float): Area under the PR curve.
        - 'lower' (float): Lower bound of the 95% confidence interval for AUC.
        - 'upper' (float): Upper bound of the 95% confidence interval for AUC.
        - 'baseline' (float): The ratio of the number of positives to the total number of samples.
    title : str, optional
        Title for the plot. Default is None.
    figsize: tuple, optional
        Default (16, 7).

    Returns
    -------
    None

    Notes
    -----
    This function plots two subplots:
    1. ROC curve with AUC and confidence interval information.
    2. Precision-Recall curve with AUC and confidence interval information.

    Each subplot includes a diagonal dashed line representing a random classifier.

    The function uses Matplotlib for plotting and displays the resulting plot.

    Examples
    --------
    >>> plot_roc_pr(roc_data, pr_data)
    """

    assert len(roc) == len(pr), 'Length of ROC and PR values do not match!'
    if isinstance(roc, dict) and isinstance(pr, dict):
        roc = (roc,)
        pr = (pr,)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    if title:
        plt.suptitle(title, fontsize=16)

    for i in range(len(roc)):
        roc_bounds = roc[i]['auc'] - roc[i]['lower']
        pr_bounds = pr[i]['auc'] - pr[i]['lower']
        if len(roc) == 1:
            roc_label = 'AUC {:.2f} 95% CI ±{:.2f}'.format(roc[i]['auc'], roc_bounds)
            pr_label = 'AUC {:.2f} 95% CI ±{:.2f}'.format(pr[i]['auc'], pr_bounds)
            colors = ['darkorange']
        else:
            roc_label = '{} | AUC {:.2f} 95% CI ±{:.2f}'.format(roc[i]['dataset'], roc[i]['auc'], roc_bounds)
            pr_label = '{} | AUC {:.2f} 95% CI ±{:.2f}'.format(pr[i]['dataset'], pr[i]['auc'], pr_bounds)
            colors = ['blue', 'green', 'red', 'yellow', 'black']

        axes[0].plot(roc[i]['fprs'], roc[i]['tprs'], color=colors[i], lw=2, label=roc_label)
        axes[1].plot(pr[i]['recalls'], pr[i]['precisions'], color=colors[i], lw=2, label=pr_label)
        axes[1].axhline(y=pr[i]['baseline'], color=colors[i], lw=2, linestyle='--')
        # Add baseline value text on the right side of the plot
        baseline_rounded = round(pr[i]['baseline'], 2)
        axes[1].text(1.02, pr[i]['baseline'], f'{baseline_rounded:.2f}',
                     color=colors[i], fontsize=10, fontweight='bold',
                     verticalalignment='center', transform=axes[1].get_yaxis_transform())

    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic')
    axes[0].legend(loc="lower right")

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall')
    axes[1].legend(loc="lower right")

    plt.show()


def plot_sidebyside(captions_dataframes):
    """
    Displays DataFrames side-by-side with formatted captions.

    Parameters
    ----------
    captions_dataframes : dict
        A dictionary where keys are captions (strings) and values are DataFrames.

    Returns
    -------
    None

    Raises
    -------
    ImportError:
        If `pandas` or `display_html` is not installed.

    Notes
    -----
    This function utilizes pandas styling and `display_html` function (likely from an
    external library) to arrange and display DataFrames with captions. It
    likely doesn't involve actual plotting functionalities.

    Example:
    -------
    >>> import pandas as pd
    >>> data1 = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    >>> data2 = pd.DataFrame({'x': [7, 8, 9], 'y': [10, 11, 12]})
    >>> captions_dict = {'data1': data1, 'data2': data2}
    >>> plot_sidebyside(captions_dict)
    """

    styles = [dict(selector="caption",
                   props=[("text-align", "center"),
                          ("font-size", "16px"),
                          ("font-weight", "bold")
                          ])]

    html_reprs = ""

    for caption, dataframe in captions_dataframes.items():
        styler = dataframe.style.set_table_attributes("style='display:inline;font-size:12px'")
        styler = styler.set_table_styles(styles)
        styler = styler.set_caption(caption)

        html_reprs = html_reprs + styler._repr_html_()

    display_html(html_reprs, raw=True)


def plot_histograms_with_cdf(hists, title='', x_start=0, figsize=(5, 5)):
    """
    Plots histograms with its cumulative distribution function (CDF) on the same graph.

    Parameters
    ----------
        hists: np.ndarray
            Array of histogram values.
        title: str
            Title of the plot. Default is "Histogram with CDF".
        x_start: int
            Starting bin of the histogram. Default is 0.
        figsize: tuple of ints
            Figure size of the histograms.
    """

    if not isinstance(hists, (list, tuple)):
        hists = [hists]  # Convert a single image to a list

    bins = np.arange(len(hists[0]) + 1)
    # Calculate the bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2

    num_hists = len(hists)
    rows = (num_hists + 2) // 3  # Calculate the number of rows needed
    cols = min(3, num_hists)  # Maximum of 4 columns

    figsize = (cols * figsize[0], rows * figsize[1])

    # Create the plot
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = np.array(axs).flatten()  # Convert the axs list to a numpy array and flatten it

    for i, hist in enumerate(hists):
        # Calculate the CDF
        cdf = np.cumsum(hist) / np.sum(hist)

        # Plot the histogram (left y-axis)
        axs[i].bar(bin_centers[x_start:], hist[x_start:],
                   width=np.diff(bins[x_start:]), edgecolor="black",
                   align="center", alpha=0.6, label="Histogram")
        axs[i].set_xlim(-1, len(bin_centers) + 100)
        axs[i].set_xlabel("Intensity Values")
        axs[i].set_ylabel("Frequency", color="blue")
        axs[i].tick_params(axis="y", labelcolor="blue")

        # Plot the CDF (right y-axis)
        ax2 = axs[i].twinx()  # Create a second y-axis
        ax2.plot(bin_centers[x_start:], cdf[x_start:], color="red", linewidth=2, label="CDF")
        ax2.set_ylabel("Cumulative Distribution", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        # Add title and legend
        fig.suptitle("{} Histogram with CDF".format(title))
        axs[i].legend(loc="upper left")
        ax2.legend(loc="upper right")

    # Remove any empty subplots
    for i in range(num_hists, rows * cols):
        fig.delaxes(axs[i])

    # Show the plot
    fig.tight_layout()
    plt.show()
