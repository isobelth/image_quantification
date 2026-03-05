from bioio import BioImage
from skimage.io import imsave
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
from skimage.transform import downscale_local_mean
import argparse
import dask.array as da


class PreProcessingDAPI:
    """Class to downsample and gaussian blur DAPI image
    Attributes:
        image_name (str): Name of the image
        image_data (numpy.ndarray): Numpy array containing the image data
        image_dim (int): Number of image dimensions
    Methods:
        load_image(self,image_path: str) -> tuple[np.ndarray, int]:
            Load the image from the specified path and return its data and number of dimensions.
        downsampling(self,img: np.ndarray, scale_factor_xy: float, scale_factor_xy: float) -> np.ndarray:
            Downsampling of the DAPI image.
        filtering(self,img: np.ndarray, sigma_xy: float, sigma_z: float) -> np.ndarray:
            Gaussian blurring of the DAPI image
        normalize(self,img: np.ndarray) -> np.ndarray:
            Normalizes the image to the range [0, 1].
        process(self,scale_factor_xy: float, scale_factor_xy: float, sigma_xy: float, sigma_z: float) -> None:
            Downsamples and gaussian blurs the DAPI image and saves it in TIFF format.
    """

    def __init__(self,
                 image_name: str,
                 image_path: str) -> None:
        self.image_name = image_name
        self.image_data,self.image_dim = self.load_image(image_path)

    def load_image(self,image_path: str) -> tuple[np.ndarray,int]:
        """Load the image from the specified path and return its data and pixel sizes.
        Args:
            image_path (str): Path to the image file.
        Returns:
            tuple: A tuple containing the image data (dask array) and image dimension.
            """
        img = BioImage(image_path)
        img_data = img.dask_data.squeeze().compute()
        img_dimensions = img_data.ndim
        return img_data,img_dimensions
    
    def downsampling(self,
                     img: np.ndarray,
                     scale_factor_xy: float,
                     scale_factor_z: float) -> np.ndarray:
        """Performs downsampling of the image.
        Args:
            img (np.ndarray): Image array to be downsampled.
            scale_factor_xy (float): Factor to downscale the image in xy dimensions.
            scale_factor_z (float): Factor to downscale the image in z dimension.
        Returns:
            np.ndarray: Downsampled image array.
        """
        scale_factor = [scale_factor_z,scale_factor_xy,scale_factor_xy] if img.ndim==3 else [scale_factor_xy,scale_factor_xy]
        return downscale_local_mean(img,factors=tuple(scale_factor))
    
    def filtering(self,
                  img:np.ndarray,
                  sigma_xy:float,
                  sigma_z:float) -> np.ndarray:
        """Applies Gaussian filtering to the image.
        Args:
            img (np.ndarray): Image array to be filtered.
            sigma_xy (float): Sigma value for Gaussian filter in xy dimensions.
            sigma_z (float): Sigma value for Gaussian filter in z dimension.
        Returns:
            np.ndarray: Filtered image array.
        """
        sigma = [sigma_z,sigma_xy,sigma_xy] if img.ndim==3 else [sigma_xy,sigma_xy]
        radius = [int(sigma_z),int(sigma_xy),int(sigma_xy)] if img.ndim==3 else [int(sigma_xy),int(sigma_xy)]
        return ndi.gaussian_filter(input=img,sigma=sigma,radius=radius)
    
    def normalize(self,img:np.ndarray) -> np.ndarray:
        """Normalizes the image to the range [0, 1].
        Args:
            img (np.ndarray): Image array to be normalized.
        Returns:
            np.ndarray: Normalized image array.
        """
        img = np.clip(img, a_min = None, a_max = np.percentile(img,99.99))
        img_min = np.min(img)
        img_max = np.max(img)
        return (img - img_min) / (img_max - img_min)
    
    def process(self,
                scale_factor_xy: float,
                scale_factor_z: float,
                sigma_xy: float,
                sigma_z: float) -> None:
        """Processes the DAPI image by downsampling and filtering.
        Args:
            scale_factor_xy (float): Factor to downscale the image in xy dimensions.
            scale_factor_z (float): Factor to downscale the image in z dimension.
            sigma_xy (float): Sigma value for Gaussian filter in xy dimensions.
            sigma_z (float): Sigma value for Gaussian filter in z dimension.
        """
        img = self.downsampling(img=self.image_data,scale_factor_xy=scale_factor_xy,scale_factor_z=scale_factor_z)
        img = self.filtering(img=img,sigma_xy=sigma_xy,sigma_z=sigma_z)
        img = self.normalize(img=img)
        path_to_save = Path.cwd()/f'{self.image_name}-dapi_processed.tif'
        print(f'Image dimension: {img.ndim}',flush=True)
        imsave(path_to_save,img,plugin='tifffile',check_contrast=False)
    

if __name__ == "__main__":
    
    # def parse arguments
    def validate_path(file_path):
        if not Path(file_path).exists():
            raise argparse.ArgumentTypeError(f"File not found: {file_path}")
        return file_path
    
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Pre process DAPI image")
        parser.add_argument("-im","--image_name",type=str,required=True,
                            help='Image name.')
        parser.add_argument("-f","--image_file", type=validate_path,required=True,
                            help='Path to image')
        parser.add_argument("--scale_factor_xy", type=int,required=True,
                            help='Factor to downscale DAPI in xy image before segmentation.')
        parser.add_argument("--scale_factor_z", type=int,required=True,
                            help='Factor to downscale DAPI in z image before segmentation.')
        parser.add_argument("--sigma_xy", type=float,required=True,
                            help='Sigma to smooth DAPI in xy image before segmentation.')
        parser.add_argument("--sigma_z", type=float,required=True,
                            help='Sigma to smooth DAPI in z image before segmentation.')
        return parser.parse_args()

    args = parse_arguments()
   
    PreProcessingDAPI(args.image_name,args.image_file).process(args.scale_factor_xy,args.scale_factor_z,args.sigma_xy,args.sigma_z)
