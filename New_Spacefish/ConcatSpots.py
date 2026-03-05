import argparse
import pandas as pd
from pathlib import Path

class ConcatSpots:
    """Class to concatenate spots from single nuclei into one table
    Attributes:
        image_name (str): Image name
        files (list): List of spot tables
    Methods:
        concat(self) -> None:
            Concatenate spots into one large table
    """

    def __init__(self,
                 image_name: str,
                 files: list[str]) -> None:
        self.image_name = image_name
        self.files = files

    def concat(self) -> None:
        """Concatenate spots into one large table"""
        df_list = [pd.read_csv(file) for file in self.files]
        df = pd.concat(df_list, ignore_index=True)
        df = df.sort_values(by=['nucleus', 'channel'])
        path_to_save = Path.cwd() / f'{self.image_name}_spots_concat.csv'
        df.to_csv(path_to_save, index=False)
        print(f'Concatenated spots are saved at {str(path_to_save)}', flush=True)

if __name__ == "__main__":

    def validate_path(file_path):
        if not Path(file_path).exists():
            raise argparse.ArgumentTypeError(f"File not found: {file_path}")
        return file_path

    def parse_arguments():
        parser = argparse.ArgumentParser(description="Concatenate CSV files")
        parser.add_argument("-im", "--image_name", type=str, required=True,
                            help='Image name.')
        parser.add_argument("-f", "--files", type=validate_path, required=True, nargs='+',
                            help='Paths to CSV files to concatenate.')
        args = parser.parse_args()
        return args

    args = parse_arguments()

    ConcatSpots(args.image_name, args.files).concat()