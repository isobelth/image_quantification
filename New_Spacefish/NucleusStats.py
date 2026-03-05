import argparse
from skimage.measure import regionprops,perimeter
from skimage.io import imread
from pathlib import Path
import numpy as np
import pandas as pd

class NucleusStats:
    """Class to calculate intensity statistics on a group of cropped nuclei.
    Attributes:
        image_name (str): Name of the image .
        files (list): List of nuclei to process.
    Methods:
        calculate_stats(self,flag: bool) -> None:
            Calculate and save nucleus intensity statistics
        write_nucleus_info(self,file: str) -> pd.DataFrame:
            Calculate single nucleus intensity statistics.
    """

    def __init__(self,
                 image_name: str,
                 files: list[str]) -> None:
        self.image_name = image_name
        self.files = files
    
    def calculate_stats(self,flag: bool) -> None:
        """Calculate and save nucleus intensity statistics"""
        df_list = [self.write_nucleus_info(file) for file in self.files]
        df = pd.concat(df_list, ignore_index=True)
        df['use_for_normalisation'] = flag
        nuclei = df['nucleus'].unique()
        path_to_save = Path.cwd()/f'{self.image_name}_{nuclei[0]}-nucleus_stats.csv'
        df.to_csv(path_to_save, index=False)

    def write_nucleus_info(self,file: str) -> pd.DataFrame:
        """Calculate single nucleus intensity statistics.
        Args:
            file (str): Path to image
        Returns:
            pd.DataFrame: Dataframe of intensity statistics
        """
        img = imread(file)
        channels = img.shape[0]-1
        img_filtered = img[:-1]
        mask = img[-1]
        nucleus_id = mask.max()
        rp = regionprops(mask.astype('uint16'))
        mask = mask == 0
        img_filtered[:, mask] = 0
        ystart,xstart,yend,xend = rp[0].bbox
        centroid = rp[0].centroid
        area = rp[0].area
        solidity = rp[0].solidity
        perimeter_value = rp[0].perimeter
        convex_hull = rp[0].convex_image
        convex_perimeter = perimeter(convex_hull)
        convexity = convex_perimeter / perimeter_value if perimeter_value > 0 else 0
        df_core = []
        df_core = pd.DataFrame({'image':[self.image_name],'nucleus':[nucleus_id],
                                'y_start':[ystart],'y_end':[yend],
                                'x_start':[xstart],'x_end':[xend],
                                'y_rel':[centroid[0]],'x_rel':[centroid[1]],
                                'area':[area],'convexity':[convexity],
                                'solidity':[solidity]})
        percentiles = np.percentile(img_filtered, [5, 25, 50, 75, 99], axis=(1, 2)).astype('float16')
        df_intensities = pd.DataFrame({
            'nucleus': [nucleus_id]*channels,
            'channel': [f'Channel{i+1}' for i in range(channels-1)]+['DAPI'],
            'intensity_min': img_filtered.min(axis=(1,2)).astype('float16'),
            'intensity_mean': img_filtered.mean(axis=(1,2)).astype('float16'),
            'intensity_std': img_filtered.std(axis=(1,2)).astype('float16'),
            'intensity_max': img_filtered.max(axis=(1,2)).astype('float16'),
            'intensity_p5': percentiles[0],'intensity_p25': percentiles[1],'intensity_p50': percentiles[2],
            'intensity_p75': percentiles[3],'intensity_p99': percentiles[4]
        })
        return pd.merge(df_core, df_intensities, on='nucleus', how='inner')
    
if __name__ == "__main__":

    # def parse arguments
    def validate_path(file_path):
        if not Path(file_path).exists():
            raise argparse.ArgumentTypeError(f"File not found: {file_path}")
        return file_path

    def parse_arguments():
        parser = argparse.ArgumentParser(description="Crop image into single nucleus files")
        parser.add_argument("-im","--image_name",type=str,required=True,
                            help='Image name.')
        parser.add_argument("-f","--files", type=validate_path,required=True,nargs="+",
                            help='Paths to cropped nuclei')
        parser.add_argument("--flag", action='store_true',
                            help='Should these nuclei be included in calculating normalisation threshold.')
        return parser.parse_args()
   
    args = parse_arguments()

    NucleusStats(args.image_name, args.files).calculate_stats(args.flag)
