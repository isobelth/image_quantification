from bioio import BioImage
from pathlib import Path
import argparse
import scipy.ndimage as ndi
from skimage.measure import regionprops
from skimage.io import imsave
import numpy as np
import pandas as pd

class SegmentationCleanUp:
    """Class to clean up segmentation masks.
    Attributes:
        image_name (str): Name of the image to be processed.
        image_data (np.ndarray): Loaded image data.
        
    Methods:
        load_image(image_path: str) -> tuple[np.ndarray,int]:
            Load the image from the specified path and return its data and pixel sizes.
        upscale(scale_factor_xy: float, scale_factor_z: float) -> None:
            Upscale the segmentation mask to the original image size.
        exclude_border_nuclei() -> None:
            Exclude nuclei that are touching the border of the image.
        remove_censored_nuclei(censor_region_path: str) -> None:
            Remove nuclei that are within the censoring region.
        save_label_mask() -> None:
            Save the current image data as a label mask.
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
    
    def upscale(self,
                scale_factor_xy: float,
                scale_factor_z:float) -> None:
        """Upscale the segmentation mask to the original image size.
        Args:
            scale_factor_xy (float): Factor to upscale the mask in xy dimensions.
            scale_factor_z (float): Factor to upscale the mask in z dimension.
        """
        scale_factor = [scale_factor_z,scale_factor_xy,scale_factor_xy] if self.image_dim==3 else [scale_factor_xy,scale_factor_xy]
        self.image_data = ndi.zoom(self.image_data,zoom=tuple(scale_factor),order=0)
        
    def save_label_mask(self) -> None:
        """Save the current image data as a label mask."""
        path_to_save = Path.cwd()/f'{self.image_name}-seg_mask_cleaned.tif'
        imsave(path_to_save,self.image_data,check_contrast=False)
        print(f'Image has been saved at this location: {str(path_to_save)}',flush=True)
    
    def exclude_border_nuclei(self) -> None:
        """Exclude nuclei that are touching the border of the image."""
        mask = self.image_data.copy()
        is3D = mask.ndim==3
        rp = regionprops(mask)
        border_nuclei = []
        ylim = np.array([0,mask.shape[1]]) if is3D else np.array([0,mask.shape[0]])
        xlim = np.array([0,mask.shape[2]]) if is3D else np.array([0,mask.shape[1]])
        for prop in rp:
            bbox = prop.bbox
            ybox = np.array([bbox[1],bbox[-2]]) if is3D else np.array([bbox[0],bbox[-2]])
            xbox = np.array([bbox[2],bbox[-1]]) if is3D else np.array([bbox[1],bbox[-1]])
            if np.any(ylim==ybox) | np.any(xlim==xbox):
                border_nuclei.append(prop.label)
        mask = np.where(np.isin(mask,border_nuclei),0,mask)
        self.image_data = mask.copy()

    def remove_censored_nuclei(self,
                               censor_region_path: str) -> None:
        """Remove nuclei that are within the censoring region.
        Args:
            censor_region_path (str): Path to the censoring region file.
        """
        if Path(censor_region_path).name == 'censor_dummy.txt':
            print('Dummy censor file provided. No nuclei will be removed.',flush=True)
        else:
            censor_region = pd.read_csv(censor_region_path)
            if list(censor_region.columns) == ['z1','z2','x_z1_1','x_z1_2','y_z1_1',
                                               'y_z1_2','x_z2_1','x_z2_2','y_z2_1','y_z2_2']:
                filtered_ids = set()
                for row in censor_region.itertuples():
                    x1 = min(row.x_z1_1,row.x_z2_1)
                    x2 = max(row.x_z1_2,row.x_z2_2)
                    y1 = min(row.y_z1_1,row.y_z2_1)
                    y2 = max(row.y_z1_2,row.y_z2_2)
                    if self.image_dim==3:
                        ids = np.unique(self.image_data[row.z1:row.z2,y1:y2,x1:x2])
                    else:
                        ids = np.unique(self.image_data[y1:y2,x1:x2])
                    filtered_ids.update(set(ids[ids>0]))
                self.image_data = np.where(np.isin(self.image_data,list(filtered_ids)),0,self.image_data)
            else:
                raise ValueError("Censor region file must have the following columns: ['z1','z2','x_z1_1','x_z1_2','y_z1_1','y_z1_2','x_z2_1','x_z2_2','y_z2_1','y_z2_2']")               


if __name__ == "__main__":
    # def parse arguments
    def validate_path(file_path):
        if not Path(file_path).exists():
            raise argparse.ArgumentTypeError(f"File not found: {file_path}")
        return file_path

    def parse_arguments():
        parser = argparse.ArgumentParser(description="Clean up segmentation")
        parser.add_argument("-im","--image_name",type=str,required=True,
                            help='Image name.')
        parser.add_argument("-f","--image_file", type=validate_path,required=True,
                            help='Path to image')
        parser.add_argument("--scale_factor_xy", type=int,required=True,
                            help='Factor to upscaling mask in xy image after segmentation.')
        parser.add_argument("--scale_factor_z", type=int,required=True,
                            help='Factor to upscaling mask in z image after segmentation.')
        parser.add_argument("--exclude_border_nuclei", action='store_true',
                            help='Whether or not to exclude nuclei at image border.')
        parser.add_argument("--censor", action='store_true',
                            help='Whether region censoring should be performed.')
        parser.add_argument("--censor_region", type=validate_path,required=False,
                            help='Path to censoring file.')
        args = parser.parse_args()
        return args

    
    args = parse_arguments()
    
    segmentationcleanup = SegmentationCleanUp(args.image_name,args.image_file)
    if args.exclude_border_nuclei:
        segmentationcleanup.exclude_border_nuclei()
    segmentationcleanup.upscale(args.scale_factor_xy,args.scale_factor_z)
    if args.censor:
        segmentationcleanup.remove_censored_nuclei(args.censor_region)
    segmentationcleanup.save_label_mask()

