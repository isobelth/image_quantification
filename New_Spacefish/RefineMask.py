import argparse
from skimage.measure import regionprops,perimeter
import skimage
from skimage.io import imsave,imread
from pathlib import Path
import numpy as np
from cellpose import models

class RefineMask:
    """Class to refine individual nucleus segmentation masks.
    Attributes:
        image_name (str): Name of the image .
        files (list): List of nuclei to process.
    Methods:
        refine_masks(self,seg_model: str,convexity_threshold: float) -> None:
            Refine segmentation masks.
        get_convexity(self,prop: skimage.measure._regionprops.RegionProperties) -> float:
            Calculate nucleus convexity
        select_central_mask(self,masks: np.ndarray,crop_size: tuple[int, int]) -> np.ndarray:
            Select the central mask of the new calculated masks
    """
    
    def __init__(self,
                 image_name: str,
                 files: list[str]) -> None:
        self.image_name = image_name
        self.files = files

    def refine_masks(self,
                    seg_model: str,
                    convexity_threshold: float) -> None:
        """Refine segmentation masks
        Args:
            seg_model (str): Name of the cellpsoe segmentation model to use
            convexity_threshold (float): Threshold to filter for nucleus convexity
        """
        seg_model = Path(seg_model)
        if seg_model.name == 'dummy':
            model = models.CellposeModel(gpu=True)
        else:
            model = models.CellposeModel(gpu=True, pretrained_model=str(seg_model))
        for file in self.files:
            img = imread(file)
            dapi = img[-2].copy()
            original_mask = img[-1].copy()
            nucleus_id = original_mask.max()
            seg_model = Path(seg_model)
            
            new_mask, flows, styles = model.eval(dapi,do_3D=False)

            new_mask = self.select_central_mask(new_mask, dapi.shape)
            new_mask = np.where(new_mask > 0, nucleus_id, 0)
        
            if np.max(new_mask) == 0:
                # Fallback to original mask
                new_mask = original_mask
            
            rp = regionprops(new_mask)
            if self.get_convexity(rp[0])>convexity_threshold:

                # Create new image stack with updated mask
                img =  np.concatenate([img[:-1], new_mask[None, :]], axis=0)

                # save crop with new mask
                path_to_save = Path.cwd()/f'{self.image_name}-nucleus{nucleus_id:04d}.tif'
                imsave(path_to_save,img,check_contrast=False)
    
    def get_convexity(self,prop: skimage.measure._regionprops.RegionProperties) -> float:
        """Calculate nucleus convexity
        Args:
            prop (skimage.measure._regionprops.RegionProperties): Nucleus region properties
        Returns:
            float: Nucleus convexity
        """
        perimeter_value = prop.perimeter
        convex_hull = prop.convex_image
        convex_perimeter = perimeter(convex_hull)
        return convex_perimeter / perimeter_value if perimeter_value > 0 else 0

    def select_central_mask(self,
                            masks: np.ndarray,
                            crop_size: tuple[int, int]) -> np.ndarray:
        """Select the central mask of the new calculated masks
        Args:
            mask (np.ndarray): segmentation mask
            crop_size (tuple): size of crop
        Returns:
            np.ndarray: new segmentation mask
        """
        if np.max(masks) == 0:
            return np.zeros_like(masks)
        center = np.array(crop_size) // 2
        regions = regionprops(masks)
        distances = [np.linalg.norm(np.array(r.centroid) - center) for r in regions]
        return np.where(masks == regions[np.argmin(distances)].label, 1, 0)
    


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
                            help='paths to images')
        parser.add_argument("-m","--segmentation_model", type=str, 
                            help='Cellpose segmentation model to use.')
        parser.add_argument("-ct","--convexity_threshold", type=float,required=False,
                            help='Threshold for filtering nuclei for convexity.')
        return parser.parse_args()

    args = parse_arguments()

    RefineMask(args.image_name, args.files).refine_masks(
        seg_model=args.segmentation_model,
        convexity_threshold=args.convexity_threshold)