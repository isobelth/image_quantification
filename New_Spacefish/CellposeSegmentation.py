import numpy as np
from skimage import io
from cellpose import models
from pathlib import Path
import argparse

class CellposeSegmentation:
    """Class to perform Cellpose segmentation on an image.
    Attributes:
        image_name (str): Name of the image to be segmented.
        image (np.ndarray): Loaded image data.
    Methods:
        load_image(self,image_path: str) -> np.ndarray:
            Load an image from the specified path.
        run_cellpose(self,seg_model: str, diameter: int, anisotropy: float,
                     do_3D: bool, min_size: int, dP_smooth: float) -> None:
            Run Cellpose segmentation on the provided image.
    """

    def __init__(self,
                 image_name: str,
                 image_path: str) -> None:
        self.image_name = image_name
        self.image = self.load_image(image_path)

    def load_image(self,
                   image_path: str) -> np.ndarray:
        """Load an image from the specified path.
        Args:
            image_path (str): Path to the image file.
        Returns:
            np.ndarray: Loaded image data.
        """
        img = io.imread(image_path)
        return img

    def run_cellpose(self, 
                     seg_model: str, 
                     anisotropy: float,
                     do_3D: bool,
                     min_size: int,
                     stitch_threshold: float) -> None:
        """Run Cellpose segmentation on the provided image.
        Args:            
            seg_model (str): The Cellpose segmentation model to use.
            anisotropy (float): Anisotropy of the image (1.0 for isotropic images).
            do_3D (bool): Whether to perform 3D segmentation.
            min_size (int): Minimum size of segmented objects.
            stitch_threshold (float): Threshold for stitching 2D masks in 3D.
        """
        # Initialize Cellpose model
        seg_model = Path(seg_model)
        if seg_model.name == 'dummy':
            model = models.CellposeModel(gpu=True)
        else:
            model = models.CellposeModel(gpu=True, pretrained_model=str(seg_model))
        
        if do_3D:
            # Run Cellpose segmentation
            masks, flows, styles = model.eval(self.image,anisotropy=anisotropy,do_3D=False,min_size=min_size,
                                    stitch_threshold=stitch_threshold,z_axis=0)
        else:
            # Run Cellpose segmentation
            masks, flows, styles = model.eval(self.image,do_3D=False,min_size=min_size)

        # Save the masks as a new file
        path_to_save = Path.cwd()/f'{self.image_name}_masks.tif'
        io.imsave(path_to_save, masks, check_contrast=False)
        
        print(f"Cellpose segmentation completed and saved to {path_to_save}")

if __name__ == "__main__":
    # def parse arguments
    def validate_path(file_path):
        if not Path(file_path).exists():
            raise argparse.ArgumentTypeError(f"File not found: {file_path}")
        return file_path

    def parse_arguments():
        parser = argparse.ArgumentParser(description="Do cellpose segmentation on an image.")
        parser.add_argument("-im","--image_name",type=str,required=True,
                            help='Image_name.')
        parser.add_argument("-f","--image_file", type=validate_path,required=True,
                            help='Path to image')
        parser.add_argument("-m","--segmentation_model", type=str, 
                            help='Cellpose segmentation model to use.')
        parser.add_argument("--anisotropy", type=float, default=1.0,
                            help='Anisotropy of the image (1.0 for isotropic images).')
        parser.add_argument("--do_3D", action='store_true',
                            help='Whether to perform 3D segmentation.')
        parser.add_argument("--min_size", type=int,
                            help='Minimum size of segmented objects.')
        parser.add_argument("--stitch_threshold", type=float,
                            help='Threshold for stitching 2D masks in 3D.')
        args = parser.parse_args()
        return args

    args = parse_arguments()


    CellposeSegmentation(args.image_name, args.image_file).run_cellpose(
        seg_model=args.segmentation_model,
        anisotropy=args.anisotropy,
        do_3D=args.do_3D,
        min_size=args.min_size,
        stitch_threshold=args.stitch_threshold)