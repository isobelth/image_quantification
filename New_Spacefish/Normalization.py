from bioio import BioImage
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from skimage.io import imsave
import re

class Normalization:
    """Class for normalise cropped nuclei and re-calculate nuclei intensity statistics
    Attributes:
        image_name (str): Name of the image
        image_path (list): List of image paths
        threshold (str): Path to threshold table
        sigma1 (float|int): Gaussian blurring for background subtraction
        sigma2 (float|int): Gaussian blurring after background subtraction
    Methods:
        load_image(self,image_path: str) -> np.ndarray:
            Load image from a file
        extract_nucleus_id(self,file: str) -> int:
            Extract nucleus ID from file name
        process_channel(self,img: np.ndarray,sigma1: float|int,sigma2: float|int) -> np.ndarray:
            Removes background (based on large gaussian blur) and clips image following gaussian smoothing
        normalize(self) -> None:
            Normalise the nucleus by removing the background a low percentile and dividing by a high percentile.
            After normalisation, intensity statistics are calculated and saved.
        write_nucleus_info(self,img: np.ndarray, nucleus_id: int) -> pd.DataFrame:
            Calculates statistics on nucleus intensities.
    """

    def __init__(self,
                 image_name: str,
                 image_path: list[str],
                 threshold: str,
                 sigma1: float|int,
                 sigma2: float|int) -> None:
        self.image_name = image_name
        self.image_path = image_path
        self.threshold = pd.read_csv(threshold)
        if 'image' in self.threshold.columns:
            self.threshold = self.threshold[self.threshold['image']==image_name].copy().reset_index(drop=True)
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def load_image(self,image_path: str) -> np.ndarray:
        """Load image from file
        Args:
            image_path (str): Path to image file.
        Returns:
            np.ndarray: Image data
        """
        img = BioImage(image_path)
        img_data = img.dask_data.squeeze().astype('float64').compute()
        return img_data
    
    def extract_nucleus_id(self,file: str) -> int:
        """Extract nucleus ID from file name
        Args:
            file (str): Path to image
        Returns:
            int: Nucleus ID
        """
        nucleus_id = Path(file).stem
        nucleus_id = nucleus_id.split('-')[-1]
        nucleus_id = re.search(r'\d+',nucleus_id).group()
        return int(nucleus_id)
    
    def process_channel(self,
                        img: np.ndarray,
                        sigma1: float|int,
                        sigma2: float|int) -> np.ndarray:
        """Removes background (based on large gaussian blur) and clips image following gaussian smoothing.
        Args:
            img (np.ndarray): Image data
            sigma1 (float|int): Sigma for calculating background
            sigma2 (float|int): Sigma for smoothing after backgroud subtraction
        Returns:
            np.ndarray: processed image
        """
        img = img-ndi.gaussian_filter(img,sigma=sigma1,radius=int(sigma1))
        img = np.clip(img,0,None)
        return ndi.gaussian_filter(img,sigma=sigma2,radius=1)
    
    def normalize(self) -> None:
        """Normalise the nucleus by removing the background a low percentile and dividing by a high percentile.
        After normalisation, intensity statistics are calculated and saved."""
        df = []
        for file in self.image_path:
            img_data = self.load_image(file)
            nucleus_id = self.extract_nucleus_id(file)
            spot_channels = img_data[:-2]
            spot_channels = np.stack([self.process_channel(channel,self.sigma1,self.sigma2) for channel in spot_channels],axis=0)
            spot_channels = spot_channels-self.threshold['intensity_p5'].values[:,None,None]
            spot_channels = np.clip(spot_channels,0,None)
            spot_channels = spot_channels/(2*self.threshold['intensity_p99'].values[:,None,None])
            dapi = (img_data[-2]/img_data[-2].max()).astype('float64')
            mask = (img_data[-1]/img_data[-1].max()).astype('float64')           
            img_stack = np.concatenate([spot_channels,dapi[np.newaxis,:,:],mask[np.newaxis,:,:]],axis=0).astype('float32')
            dtype = (2**16)-1
            img_stack = (np.clip(img_stack,0,1)*dtype).astype('uint16')
            path_to_save = Path.cwd()/f'{self.image_name}-nucleus{nucleus_id:04d}-normalised.tif'
            imsave(path_to_save,img_stack,check_contrast=False)
            print(f'{self.image_name}-{nucleus_id} was saved at {str(path_to_save)}.')
            df.append(self.write_nucleus_info(img_stack,nucleus_id))
        path_to_save = Path.cwd()/f'nuclei_stats_normalized_nucleus{nucleus_id}.csv'
        pd.concat(df).to_csv(path_to_save,index=False)
        print(f'Statistics on nuclei are saved at {str(path_to_save)}',flush=True)

    def write_nucleus_info(self,
                           img: np.ndarray,
                           nucleus_id: int) -> pd.DataFrame:
        """Calculates statistics on nucleus intensities.
        Args:
            img (np.ndarray): Image data
            nucleus_id (int): Nucleus ID
        Returns:
            pd.DataFrame: Table of intensity statistics.
        """
        img_dim = img.ndim
        img_shape = img.shape[0]-1
        img_filtered = img[:-1]
        mask = img[-1] == 0
        img_filtered[:, mask] = 0
        percentiles = np.percentile(img_filtered, [5, 25, 50, 75, 99], axis=(1, 2, 3) if img_dim == 4 else (1, 2)).astype('float32')
        df_intensities = pd.DataFrame({
            'image': [self.image_name]*img_shape,
            'nucleus': [nucleus_id]*img_shape,
            'channel': [f'Channel{i+1}' for i in range(img_shape-1)]+['DAPI'],
            'intensity_min': img_filtered.min(axis=(1,2,3) if img_dim==4 else (1,2)).astype('float16'),
            'intensity_mean': img_filtered.mean(axis=(1,2,3) if img_dim==4 else (1,2)).astype('float16'),
            'intensity_std': img_filtered.std(axis=(1,2,3) if img_dim==4 else (1,2)).astype('float16'),
            'intensity_max': img_filtered.max(axis=(1,2,3) if img_dim==4 else (1,2)).astype('float32'),
            'intensity_p5': percentiles[0],'intensity_p25': percentiles[1],'intensity_p50': percentiles[2],
            'intensity_p75': percentiles[3],'intensity_p99': percentiles[4]
        })
        return df_intensities
  
if __name__ == '__main__':
    # def parse arguments
    def validate_path(file_path):
        if not Path(file_path).exists():
            raise argparse.ArgumentTypeError(f"File not found: {file_path}")
        return file_path


    def parse_arguments():
        parser = argparse.ArgumentParser(description="Normalize nuclei")
        parser.add_argument("-im","--image_name",type=str,required=True,
                            help='Image name.')
        parser.add_argument("-f","--image_file", type=validate_path,required=True,nargs='+',
                            help='Path to image')
        parser.add_argument("-t","--threshold", type=validate_path,required=True,
                            help='Path to thresholds')
        parser.add_argument("--sigma1", type=float,required=True,
                        help='Sigma value for background.')
        parser.add_argument("--sigma2", type=float,required=True,
                            help='Sigma to smooth.')
        args = parser.parse_args()
        if args.sigma1<=0 or args.sigma2<=0:
            parser.error("--sigma1 and --sigma2 have to be greater than 0.")
        return args


    args = parse_arguments()

    Normalization(args.image_name,args.image_file,args.threshold,args.sigma1,args.sigma2).normalize()