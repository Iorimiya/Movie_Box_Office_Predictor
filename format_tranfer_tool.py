import re
import json
from pathlib import Path
from datetime import datetime
import yaml
from movie_data import MovieData


class FormatTransferTool:
    def __init__(self, input_folder_path: str, output_file_path: str):
        self.input_file_list = list()
        self.input_file_suffix = '.json'
        self.output_file_path = Path(output_file_path)
        self.output_file_suffix = '.yaml'
        self.input_encoding = 'utf-8-sig'
        self.output_encoding = 'utf-8'

        self.input_file_list = Path(input_folder_path).glob(f"*{self.input_file_suffix}")

    def print_all_input_file(self):
        [print(input_file) for input_file in self.input_file_list]
        return

    def read_json_data(self, json_file_path: Path) -> MovieData | None:
        date_format = '%Y-%m-%d'
        date_split_pattern = '~'
        with open(json_file_path, mode='r', encoding=self.input_encoding) as file:
            json_data = json.load(file)
        try:
            weekly_box_office_data = [{
                'start_date': datetime.strptime(re.split(date_split_pattern, week_data["Date"])[0],
                                                date_format).date(),
                'end_date': datetime.strptime(re.split(date_split_pattern, week_data["Date"])[1],
                                              date_format).date(),
                'box_office': int(week_data["Amount"])} for week_data in json_data['Rows']]
        except TypeError:
            return None
        if not weekly_box_office_data:
            return None
        return MovieData(movie_name=json_file_path.stem, box_office=weekly_box_office_data)

    def write_to_yaml_file(self, data: list[MovieData]):
        yaml.Dumper.ignore_aliases = lambda self, data: True
        with open(self.output_file_path, mode='w', encoding=self.output_encoding) as file:
            yaml.dump(data, file, allow_unicode=True)

    def transfer_data(self):
        movie_data_list: list[MovieData] = [self.read_json_data(input_file) for input_file in self.input_file_list]
        self.write_to_yaml_file(movie_data_list)


if __name__ == '__main__':
    input_path = 'data/weekly_box_office_data/by_movie_name'
    output_path = 'data/weekly_box_office_data/all_data.yaml'
    ftt = FormatTransferTool(input_path, output_path)
    ftt.transfer_data()
