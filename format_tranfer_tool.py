import csv
import json
from pathlib import Path


class FormatTransferTool:
    def __init__(self, input_folder_path: str, output_file_path: str):
        self.input_file_list = list()
        self.input_file_suffix = '.json'
        self.output_file_path = Path(output_file_path)
        self.output_file_suffix = '.csv'
        self.encoding = 'utf-8'

        self.input_file_list = Path(input_folder_path).glob(f"*.{self.input_file_suffix}")

    def print_all_input_file(self):
        [print(input_file) for input_file in self.input_file_list]
        return

    def read_json_data(self,json_file_path:Path):
        with open(json_file_path, mode='r', encoding=self.encoding)as file:
            return json.load(file)

    def write_to_csv_file(self,csv_file_path:Path):
        with open(csv_file_path, mode='w', encoding=self.encoding)as file:
            pass


if __name__ == '__main__':
    input_file_path = ',/data/weekly_box_office_data/by_movie_name/音爆浩劫.json'
    output_file_path = ''
