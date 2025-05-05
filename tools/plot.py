import re
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from tools.constant import Constants
from machine_learning_model.box_office_prediction import MoviePredictionModel

def plot_line_graph(title: str, save_file_path: Path,
                    x_data: list[int], y_data: list[float],
                    format_type:str, y_label: str,
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
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    match format_type:
        case 'percent':
            plt.gca().yaxis.set_major_formatter(PercentFormatter())
        case 'sci-notation':
            plt.gca().ticklabel_format(style='sci', scilimits=(-2, 1), axis='y')
        case _:
            raise ValueError(f'Format need to be either \'percent\' or \'sci-notation\'.')
    plt.plot(x_data, y_data)
    plt.savefig(save_file_path)
    plt.show()
    return


def plot_loss(log_path: Path) -> None:
    """
    Loading logs and draw line graph of training validation loss value.

    Args:
        log_path: The log containing the training validation loss value.

    Returns:
        None.
    """
    # read log content
    with open(log_path, 'r') as file:
        text: str = file.read()
    # find loss value in every saving epoch and calculate epoch.
    final_epoch_search_pattern: Final[str] = 'INFO - epoch inputted: \d+$'
    step_epoch_search_pattern: Final[str] = 'INFO - loop epoch inputted: \d+$'
    final_epoch: int = int(re.search(final_epoch_search_pattern, text, re.MULTILINE).group(0).rsplit(': ')[-1])
    step_epoch: int = int(re.search(step_epoch_search_pattern, text, re.MULTILINE).group(0).rsplit(': ')[-1])
    model_information_search_pattern: Final[
        str] = '^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] \{[^:]+:\d+} INFO - Epoch \d+: Training Loss = .+\n\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] \{[^:]+:\d+} INFO - model test loss: .+\.$'
    found_information: list = re.findall(model_information_search_pattern, text, re.MULTILINE)
    init_epoch = final_epoch - len(found_information) * step_epoch

    model_epochs: list[int] = [epoch + step_epoch for single_record, epoch in
                               zip(found_information, range(init_epoch, final_epoch, step_epoch))]
    model_losses: list[float] = [
        float(re.search('loss: .+\.', single_record).group(0).rsplit(' ')[-1].rsplit('.', 1)[0]) for
        single_record, epoch in zip(found_information, range(init_epoch, final_epoch, step_epoch))]

    # pyplot drawing
    plot_line_graph(title='training_validation_loss', save_file_path=Path('graph/training_validation_loss.png'),
                    x_data=model_epochs, y_data=model_losses,
                    format_type='percent', y_label='loss')

    return


def plot_trend(model_name: str):
    """
    Plots the trend accuracy of a specified model across different training epochs.

    Args:
        model_name (str): The name of the model to evaluate.

    Returns:
        None.
    """
    folder_list: list[Path] = list(Constants.BOX_OFFICE_PREDICTION_FOLDER.glob(f"{model_name}_*"))

    model_epochs: list[int] = [int(folder.name.split("_")[-1]) for folder in folder_list]
    accuracies: list[float] = [MoviePredictionModel(model_path=folder.joinpath(f"{folder.name}.keras"),
                                                    training_setting_path=folder.joinpath(f'setting.yaml'),
                                                    transform_scaler_path=folder.joinpath(f'scaler.gz')) \
                                   .evaluate_trend(test_data_folder_path=folder) for folder in folder_list]
    model_epochs, accuracies = (zip(*sorted(zip(model_epochs, accuracies), key=lambda x: x[0])))

    # pyplot drawing
    plot_line_graph(title='trend_accuracy', save_file_path=Path('../graph/trend_accuracy.png'),
                    x_data=model_epochs, y_data=accuracies,
                    format_type='percent', y_label='accuracy')


def plot_range(model_name: str):
    """
    Plots the range accuracy of a specified model across different training epochs.

    Args:
        model_name (str): The name of the model to evaluate.

    Returns:
        None.
    """
    folder_list: list[Path] = list(Constants.BOX_OFFICE_PREDICTION_FOLDER.glob(f"{model_name}_*"))

    model_epochs: list[int] = [int(folder.name.split("_")[-1]) for folder in folder_list]
    accuracies: list[float] = [MoviePredictionModel(model_path=folder.joinpath(f"{folder.name}.keras"),
                                                    training_setting_path=folder.joinpath(f'setting.yaml'),
                                                    transform_scaler_path=folder.joinpath(f'scaler.gz')) \
                                   .evaluate_range(test_data_folder_path=folder) for folder in folder_list]
    model_epochs, accuracies = (zip(*sorted(zip(model_epochs, accuracies), key=lambda x: x[0])))

    plot_line_graph(title='range_accuracy', save_file_path=Path('../graph/range_accuracy.png'),
                    x_data=model_epochs, y_data=accuracies,
                    format_type='percent', y_label='accuracy')
