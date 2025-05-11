import re
import logging
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from tools.constant import Constants
from machine_learning_model.box_office_prediction import MoviePredictionModel


def search_model(model_name: str) -> tuple[list[Path], list[int]]:
    logging.info(f"Search models in \"{Constants.BOX_OFFICE_PREDICTION_FOLDER}\" folder")
    folder_list: list[Path] = list(
        filter(lambda file: file.is_dir(), Constants.BOX_OFFICE_PREDICTION_FOLDER.glob(f"{model_name}_*")))
    model_epochs: list[int] = [int(folder.name.split("_")[-1]) for folder in folder_list]
    return folder_list, model_epochs


def plot_line_graph(title: str, save_file_path: Path,
                    x_data: list[int], y_data: list[float],
                    format_type: str, y_label: str,
                    x_label: str = 'epoch') -> None:
    """
    Plots a line graph with specified title, data, and formatting.

    Args:
        title (str): The title of the graph.
        save_file_path (Path): The path to save the graph image.
        x_data (list[int]): The data for the x-axis.
        y_data (list[float]): The data for the y-axis.
        format_type (str): The formatting type for the y-axis ('percent' or 'sci-notation').
        y_label (str): The label for the y-axis.
        x_label (str): The label for the x-axis (default: 'epoch').

    Raises:
        ValueError: If `format_type` is not 'percent' or 'sci-notation'.
    """
    logging.info("Plot line graph.")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    match format_type:
        case 'percent':
            logging.info("Plot line graph with percent formatting.")
            plt.gca().yaxis.set_major_formatter(PercentFormatter())
        case 'sci-notation':
            logging.info("Plot line graph with sci-notation formatting.")
            plt.gca().ticklabel_format(style='sci', scilimits=(-2, 1), axis='y')
        case _:
            raise ValueError(f"Format need to be either \'percent\' or \'sci-notation\'.")
    plt.plot(x_data, y_data)
    logging.info(f"Saving image to \"{save_file_path}\".")
    plt.savefig(save_file_path)
    plt.show()
    return


def plot_training_loss(log_path: Path) -> None:
    """
    Loading logs and draw line graph of training validation loss value.

    Args:
        log_path: The log containing the training validation loss value.

    Returns:
        None.
    """
    logging.info("Plot line graph of training loss.")
    # read log content
    logging.info(f"Read log file from \"{log_path}\".")
    with open(log_path, 'r') as file:
        text: str = file.read()

    logging.info("Collect loss value form log content.")
    # find loss value in every saving epoch and calculate epoch.
    target_epoch_search_pattern: Final[str] = 'INFO - (Target )?epoch inputted: \d+\.?$'
    target_epoch: int = int(list(
        filter(lambda x: x, re.split(": |\.?$", re.search(target_epoch_search_pattern, text, re.MULTILINE).group(0))))[
                                -1])
    logging.info(f"Found target epoch: {target_epoch}.")
    saving_interval_search_pattern: Final[list[str]] = ['INFO - Saving model every \d+ epoch.$',
                                                        'INFO - loop epoch inputted: \d+\.?$']
    try:
        saving_interval: int = int(
            re.split(' ', re.search(saving_interval_search_pattern[0], text, re.MULTILINE).group(0))[-2])
    except AttributeError:
        saving_interval: int = int(
            re.split(' ', re.search(saving_interval_search_pattern[1], text, re.MULTILINE).group(0))[-1])

    logging.info(f"Found saving interval of epoch: {saving_interval}.")
    model_information_search_pattern: Final[str] = \
        "^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] \{[^:]+:\d+} INFO - Epoch \d+: Training (?:L|l)oss = [\de\+\-\.]+\.?\n" \
        "\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] \{[^:]+:\d+} INFO - (?:Model validation|model test) loss: [\de\+\-\.]+\.?$"
    found_information: list = re.findall(model_information_search_pattern, text, re.MULTILINE)
    init_epoch = target_epoch - len(found_information) * saving_interval
    logging.info(f"Epoch when start training: {init_epoch}.")

    model_epochs: list[int] = [epoch + saving_interval for single_record, epoch in
                               zip(found_information, range(init_epoch, target_epoch, saving_interval))]
    loss_search_pattern: Final[str] = '(L|l)oss = .+'

    model_losses: list[float] = [
        float(list(filter(lambda x: x, re.split(" |\.?$", re.search(loss_search_pattern, single_record).group(0))))[-1])
        for
        single_record, epoch in zip(found_information, range(init_epoch, target_epoch, saving_interval))]

    # pyplot drawing
    plot_line_graph(title='training_loss', save_file_path=Path('data/graph/training_loss.png'),
                    x_data=model_epochs, y_data=model_losses,
                    format_type='sci-notation', y_label='loss')

    return


def plot_validation_loss(model_name: str) -> None:
    """
    Plots the test validation loss of a specified model across different training epochs.

    Args:
        model_name (str): The name of the model to evaluate.

    Returns:
        None.
    """
    logging.info("Plot line graph of validation loss.")
    folder_list, model_epochs = search_model(model_name=model_name)
    logging.info("calculating validation loss.")
    loss: list[float] = [MoviePredictionModel(model_path=folder.joinpath(f"{folder.name}.keras"),
                                              training_setting_path=folder.joinpath(f'setting.yaml'),
                                              transform_scaler_path=folder.joinpath(f'scaler.gz')) \
                             .evaluate_loss(test_data_folder_path=folder) for folder in folder_list]
    model_epochs, loss = (zip(*sorted(zip(model_epochs, loss), key=lambda x: x[0])))

    # pyplot drawing
    plot_line_graph(title='validation_loss', save_file_path=Path('data/graph/validation_loss.png'),
                    x_data=model_epochs, y_data=loss,
                    format_type='sci-notation', y_label='loss')


def plot_trend_accuracy(model_name: str):
    """
    Plots the trend accuracy of a specified model across different training epochs.

    Args:
        model_name (str): The name of the model to evaluate.

    Returns:
        None.
    """
    logging.info("Plot line graph of trend accuracy.")
    folder_list, model_epochs = search_model(model_name=model_name)
    logging.info("calculating trend accuracy.")
    accuracies: list[float] = [MoviePredictionModel(model_path=folder.joinpath(f"{folder.name}.keras"),
                                                    training_setting_path=folder.joinpath(f'setting.yaml'),
                                                    transform_scaler_path=folder.joinpath(f'scaler.gz')) \
                                   .evaluate_trend(test_data_folder_path=folder) for folder in folder_list]
    model_epochs, accuracies = (zip(*sorted(zip(model_epochs, accuracies), key=lambda x: x[0])))

    # pyplot drawing
    plot_line_graph(title='trend_accuracy', save_file_path=Path('data/graph/trend_accuracy.png'),
                    x_data=model_epochs, y_data=list(map(lambda accuracy: 100 * accuracy, accuracies)),
                    format_type='percent', y_label='accuracy')


def plot_range_accuracy(model_name: str):
    """
    Plots the range accuracy of a specified model across different training epochs.

    Args:
        model_name (str): The name of the model to evaluate.

    Returns:
        None.
    """
    logging.info("Plot line graph of range accuracy.")
    folder_list, model_epochs = search_model(model_name=model_name)
    logging.info("calculating range accuracy.")
    accuracies: list[float] = [MoviePredictionModel(model_path=folder.joinpath(f"{folder.name}.keras"),
                                                    training_setting_path=folder.joinpath(f'setting.yaml'),
                                                    transform_scaler_path=folder.joinpath(f'scaler.gz')) \
                                   .evaluate_range(test_data_folder_path=folder) for folder in folder_list]
    model_epochs, accuracies = (zip(*sorted(zip(model_epochs, accuracies), key=lambda x: x[0])))

    plot_line_graph(title='range_accuracy', save_file_path=Path('data/graph/range_accuracy.png'),
                    x_data=model_epochs, y_data=list(map(lambda accuracy: 100 * accuracy, accuracies)),
                    format_type='percent', y_label='accuracy')
