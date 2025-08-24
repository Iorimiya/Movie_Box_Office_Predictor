from logging import Logger
from pathlib import Path
from typing import Literal, Optional, TypedDict

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from src.core.logging_manager import LoggingManager


class PlotDataset(TypedDict):
    """
    Represents a single dataset to be plotted on a graph.

    :ivar label: The label for this dataset, which will appear in the legend.
    :ivar data: A list of numerical values for the y-axis.
    """
    label: str
    data: list[float]


def plot_multi_line_graph(
    title: str,
    save_path: Path,
    x_data: list[int | float],
    y_datasets: list[PlotDataset],
    x_label: str,
    y_label: str,
    y_formatter: Optional[Literal['percent', 'sci-notation']] = None
) -> None:
    """
    Plots a line graph with one or more datasets, a legend, and saves it.

    :param title: The title of the graph.
    :param save_path: The full path where the graph image will be saved.
    :param x_data: The data for the x-axis, shared by all y-datasets.
    :param y_datasets: A list of PlotDataset dictionaries, each representing a line to plot.
    :param x_label: The label for the x-axis.
    :param y_label: The label for the y-axis.
    :param y_formatter: Optional formatting for the y-axis ('percent' or 'sci-notation').
    """
    logger: Logger = LoggingManager().get_logger('root')
    logger.info(f"Plotting multi-line graph: '{title}'")

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    for dataset in y_datasets:
        plt.plot(x_data, dataset['data'], label=dataset['label'])

    if y_formatter:
        match y_formatter:
            case 'percent':
                logger.info("Applying 'percent' formatting to y-axis.")
                plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=100))
            case 'sci-notation':
                logger.info("Applying 'sci-notation' formatting to y-axis.")
                plt.gca().ticklabel_format(style='sci', scilimits=(-2, 3), axis='y')
            case _:
                # This case should ideally not be hit if types are checked, but as a safeguard.
                logger.warning(f"Unknown y_formatter '{y_formatter}'. No formatting applied.")

    plt.legend()
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving graph to: '{save_path}'")
    plt.savefig(save_path)
    plt.show()
    plt.close()
