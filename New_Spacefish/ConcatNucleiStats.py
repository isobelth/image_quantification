import argparse
import pandas as pd
from pathlib import Path

class ConcatCSV:
    """Class to concatenate nucleus statistics files.
    Attributes:
        image_name (str): Name of the image .
        files (list): List of statistic files.
    Methods:
        concat(self) -> None:
            Calculate and save nucleus intensity statistics
        write_nucleus_info(self,file: str) -> pd.DataFrame:
            Calculate single nucleus intensity statistics.
    """

    def __init__(self,
                 image_name: str,
                 files: list[str]) -> None:
        self.image_name = image_name
        self.files = files

    def concat(self,
               datatype: str) -> None:
        """Concat csv files.
        Args:
            datatype (str): Whether these are the raw or normalised data
        """
        df_list = [pd.read_csv(file) for file in self.files]
        df = pd.concat(df_list, ignore_index=True)
        df = df.sort_values(by=['nucleus', 'channel'])
        if datatype == 'raw':
            path_to_save = Path.cwd() / f'{self.image_name}_NucleiStats.csv'
        else:
            path_to_save = Path.cwd() / f'{self.image_name}_NucleiStats_normalized.csv'
        df.to_csv(path_to_save, index=False)
        print(f'Statistics on nuclei are saved at {str(path_to_save)}', flush=True)

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
        parser.add_argument("--datatype",type=str, required=True,
                            help='Decision of data is raw or normalised.')
        args = parser.parse_args()
        return args

    args = parse_arguments()

    ConcatCSV(args.image_name, args.files).concat(args.datatype)