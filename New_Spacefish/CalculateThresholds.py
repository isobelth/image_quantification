import pandas as pd
import argparse
from pathlib import Path


class CalculateThreshold:
    """Class to calculate the threshold for normalisation.
    Attributes:
        nuclei_stats (list): List of paths to nucleus statistics
    Methods:
        load_csv(self,files: list[str]) -> pd.DataFrame:
            Load nucleus statistics
        calculate_thresholds(self) -> None:
            Calculate thresholds for intensity normalization.
    """

    def __init__(self,nuclei_stats: list[str]):
        self.nuclei_stats = self.load_csv(nuclei_stats)

    def load_csv(self,files: list[str]) -> pd.DataFrame:
        """Load nucleus statistics
        Args:
            files (list): List of paths to nucleus statistics
        Returns:
            pd.DataFrame: Dataframe of all nucleus statistics
        """
        if len(files)>1:
            df = [pd.read_csv(file) for file in files]
            return pd.concat(df,ignore_index=True)
        else:
            return pd.read_csv(files[0])
            
    def calculate_thresholds(self,default_p99: int|float,global_nucleus_normalisation: bool) -> None:
        """Calculate thresholds for intensity normalization.
        Args:
            default_p99 (int|float): Default p99 value to check calculated value with.
            global_nucleus_normalisation (bool): Whether to perform global normalization across all images.
        """
        # calculate nucleus thresholds
        if global_nucleus_normalisation:
            self.nuclei_stats = self.nuclei_stats[self.nuclei_stats['use_for_normalisation']].copy().reset_index(drop=True)
            nucleus_threshold = self.nuclei_stats.groupby('channel').agg({'intensity_p5':'mean',
                                                                      'intensity_p99': lambda x: max(x.mean(),default_p99)}).reset_index()
        else:
            nucleus_threshold = self.nuclei_stats.groupby(['image','channel']).agg({'intensity_p5':'mean',
                                                                      'intensity_p99': lambda x: max(x.mean(),default_p99)}).reset_index()     
        nucleus_threshold = nucleus_threshold[nucleus_threshold['channel'] != 'DAPI']
        # sort values by channel
        nucleus_threshold = nucleus_threshold.sort_values(by='channel').reset_index(drop=True) if global_nucleus_normalisation else nucleus_threshold.sort_values(by=['image','channel']).reset_index(drop=True)
        # save file
        path_to_save = Path.cwd()/'normalization_threshold.csv'
        nucleus_threshold.to_csv(path_to_save,index=False)
        print(f'Threshold for normalization was saved at {str(path_to_save)}.')


if __name__ == '__main__':
    # def parse arguments
    def validate_path(file_path):
        if not Path(file_path).exists():
            raise argparse.ArgumentTypeError(f"File not found: {file_path}")
        return file_path

    def parse_arguments():
        parser = argparse.ArgumentParser(description="Calculate threshold for normalization")
        parser.add_argument("--nuclei_stats",type=validate_path,required=True,nargs="+",
                            help='Path to nuclei stats.')
        parser.add_argument("--default_p99",type=float,required=True,
                            help='Default p99 value to check against.')
        parser.add_argument("--global_nucleus_normalisation", action='store_true',
                            help='Whether to perform global normalization across all images.')
        return parser.parse_args()

    args = parse_arguments()

    CalculateThreshold(args.nuclei_stats).calculate_thresholds(args.default_p99,args.global_nucleus_normalisation)
    