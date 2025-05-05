from pathlib import Path
import re
from typing import Final

import matplotlib.pyplot as plt


def plot_loss(log_path: Path) -> None:
    """

    :return:
    """
    with open(log_path, 'r') as file:
        text:str = file.read()
        final_epoch_search_pattern:Final[str] = 'INFO - epoch inputted: \d+$'
        step_epoch_search_pattern:Final[str] = 'INFO - loop epoch inputted: \d+$'
        final_epoch: int = int(re.search(final_epoch_search_pattern, text, re.MULTILINE).group(0).rsplit(': ')[-1])
        step_epoch: int = int(re.search(step_epoch_search_pattern, text, re.MULTILINE).group(0).rsplit(': ')[-1])
        model_information_search_pattern: Final[str] = '^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] \{[^:]+:\d+} INFO - Epoch \d+: Training Loss = .+\n\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] \{[^:]+:\d+} INFO - model test loss: .+\.$'
        found_informations: list = re.findall(model_information_search_pattern, text, re.MULTILINE)
        init_epoch = final_epoch - len(found_informations) * step_epoch

        model_epochs: list[int] = [epoch + step_epoch for single_record, epoch in zip(found_informations, range(init_epoch, final_epoch, step_epoch))]
        model_losses:list[float] = [float(re.search('loss: .+\.', single_record).group(0).rsplit(' ')[-1].rsplit('.', 1)[0]) for single_record, epoch in zip(found_informations, range(init_epoch, final_epoch, step_epoch))]
        print(model_epochs, model_losses)


if __name__ == '__main__':
    plot_loss(Path('log/2025-05-04T14：58：49_INFO.log'))
