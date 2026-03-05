import argparse
import pandas as pd
import numpy as np
from pathlib import Path


class CountTable:
    """Class to create a gene count matrix
    Attributes
        image_name (str): Image name
        decoded_spots (pd.DataFrame): Table of decoded barcodes
        genes (list): List of gene names
        nucleus_stats (pd.DataFrame): List of nuclei and their intensity statistics
    Methods:
        get_genes(self,file: str) -> list[str]:
            Read genes from the codebook file
        create_table(self) -> None:
            Create gene count matrix and save it as file.
    """

    def __init__(self,
                 image_name: str,
                 decoded_spots: str,
                 codebook: str,
                 nucleus_stats: str) -> None:
        self.image_name = image_name
        self.decoded_spots = pd.read_csv(decoded_spots)
        self.genes = self.get_genes(codebook)
        self.nucleus_stats = pd.read_csv(nucleus_stats)

    def get_genes(self,file: str) -> list[str]:
        """Read genes from the codebook file
        Args:
            file (str): Path to the codebook
        Returns:
            list: List of gene names
        """
        df = pd.read_csv(file)
        return df['gene'].to_list()
    
    def create_table(self) -> None:
        """Create gene count matrix and save it as file."""
        count_table = self.decoded_spots.groupby(['nucleus','gene']).size().unstack(fill_value=0)
        count_table = count_table.reindex(columns=self.genes,fill_value=0)
        count_table = count_table.reindex(index=sorted(np.unique(self.nucleus_stats['nucleus'])),fill_value=0).reset_index()
        count_table.insert(0,column='name',value=self.image_name)
        path_to_save = Path.cwd()/f'{self.image_name}_count_table.csv'
        count_table.to_csv(path_to_save,index=False)


if __name__ == '__main__':
    # def parse arguments
    def validate_path(file_path):
        if not Path(file_path).exists():
            raise argparse.ArgumentTypeError(f"File not found: {file_path}")
        return file_path


    def parse_arguments():
        parser = argparse.ArgumentParser(description="Create results table")
        parser.add_argument("-im","--image_name",type=str,required=True,
                            help='Image name.')
        parser.add_argument("-ds","--decoded_spots", type=validate_path,required=True,
                            help='Path to decoded spots')
        parser.add_argument("-c","--codebook", type=validate_path,required=True,
                            help='Path to codebook')
        parser.add_argument("--nucleus_stats", type=validate_path,required=True,
                            help='Path to nucleus stats')
        args = parser.parse_args()
        return args

    args = parse_arguments()

    CountTable(args.image_name,args.decoded_spots,args.codebook,args.nucleus_stats).create_table()
