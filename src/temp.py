from pathlib import Path
from src.data_handling.file_io import CsvFile
from src.data_handling.database_client import DatabaseConfig
from src.data_handling.dataset import Dataset

if __name__ == '__main__':
    user_config = DatabaseConfig(address='localhost', port='27045', user='mbop_user', password='mbop_pass')
    root_config = DatabaseConfig(address='localhost', port='27045', user='root', password='root')

    db = Dataset(name='test1',mode='DATABASE', override_database_config=user_config)
    # db.initialize_dataset(source_csv=CsvFile(path=Path('./inputs/raw_index_sources/test.csv')),root_config=root_config)
    # db.collect_box_office()
    db.collect_public_review(target_website='PTT')
