import csv
import json
from pathlib import Path


class FormatTransferTool:
    def __init__(self, input_folder_path: str, output_file_path: str):
        self.input_file_list = list()
        self.input_file_suffix = '.json'
        self.output_file_path = Path(output_file_path)
        self.output_file_suffix = '.csv'
        self.input_encoding = 'utf-8-sig'
        self.output_encoding = 'utf-8'

        self.input_file_list = Path(input_folder_path).glob(f"*{self.input_file_suffix}")

    def print_all_input_file(self):
        [print(input_file) for input_file in self.input_file_list]
        return

    def read_json_data(self, json_file_path: Path):
        with open(json_file_path, mode='r', encoding=self.input_encoding) as file:
            return json.load(file)

    def write_to_csv_file(self, csv_file_path: Path):
        with open(csv_file_path, mode='w', encoding=self.output_encoding) as file:
            pass

    def transfer_data(self):
        json_data_list = [self.read_json_data(input_file) for input_file in self.input_file_list]
        for i,json_data in enumerate(json_data_list):
            for week_data in json_data['Rows']:
                try:
                    print(f'{i},{week_data["Date"]},{int(week_data["Amount"])}')
                except TypeError:
                    break


if __name__ == '__main__':
    input_path = 'data/weekly_box_office_data/by_movie_name'
    output_path = ''
    ftt = FormatTransferTool(input_path, output_path)
    ftt.transfer_data()
