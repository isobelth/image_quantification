from bioio import BioImage
from bioio.writers import OmeZarrWriter
from skimage.io import imsave
import bioio_base
from pathlib import Path
import argparse
import numpy as np
import dask.array as da

class SplitImage:
    """Class to split an image into different channels and save them in OME Zarr format.
    During this process, the image gets reshaped to match the requirements of later preprocessing steps.
    Attributes:
        image_name (str): Name of the image.
        image_data (dask.array.Array): Dask array containing the image data.
        pixel_size (bioio_base.types.PhysicalPixelSizes): Physical pixel sizes of the image
    Methods:
        load_image(self,image_path: str,scene: int) -> tuple[da.Array, bioio_base.types.PhysicalPixelSizes]:
            Load the image from the specified path and return its data and pixel sizes.
        reshape_image(self,do_z_projection: bool, scaling_factor: tuple[float, float, float], anisotropy: float) -> None:
            Reshape image dimensions (and z project) to match later preprocessing steps.
        save_spot_channels(self,index: int) -> None:
            Save the spot channels of the image data in OME Zarr format.
        save_dapi(self,index: int) -> None:
            Save the DAPI channel of the image data in OME Zarr format.
        save_brightfield(self,index: int) -> None:
            Save the brightfield channel of the image data in TIFF format.
    """
    
    def __init__(self, image_name: str,
                 image_path: str,
                 scene: int) -> None:
        self.image_name = image_name
        self.image_data, self.pixel_size = self.load_image(image_path,scene)

    def load_image(self,image_path: str,scene: int) -> tuple[da.Array,bioio_base.types.PhysicalPixelSizes]:
        """Load the image from the specified path and return its data and pixel sizes.
        Args:
            image_path (str): Path to the image file.
            scene (int): Index of the scene to load.
        Returns:
            tuple: A tuple containing the image data (dask array) and pixel sizes.
            """
        img = BioImage(image_path)
        img.set_scene(scene)
        pixel_size = img.physical_pixel_sizes
        img_data = img.dask_data.squeeze()
        return img_data, pixel_size
    
    def reshape_image(self,
                      do_z_projection: bool,
                      scaling_factor: tuple[float,float,float],
                      anisotropy: float) -> None:
        """Reshape image dimensions (and z project) to match later preprocessing steps.
        Args:
            do_z_projection (bool): Whether to perform z-projection on the image data.
            scaling_factor (tuple[float,float,float]): Scaling factors for the image dimensions.
            anisotropy (float): Anisotropy factor for the image in z dimension.
        """
        shape = np.array(self.image_data.shape[1:])
        new_shape = (np.ceil(shape/scaling_factor)*scaling_factor).astype('int')
        zpad,ypad,xpad = new_shape-shape
        if do_z_projection:
            self.image_data = da.max(self.image_data,axis=1)
            if np.sum(shape%np.array(scaling_factor))!=0:
                self.image_data = da.pad(self.image_data,pad_width=((0,0),(0,ypad),(0,xpad)),mode='constant',constant_values=0)
            self.pixel_size = bioio_base.types.PhysicalPixelSizes(Z=None,Y=self.pixel_size[1],X=self.pixel_size[2])
        else:
            if (shape[0]/scaling_factor[0])<=anisotropy:
                raise ValueError(
                f'Provided scaling factor ({scaling_factor}) results in fewer z slices ({shape[0]/scaling_factor[0]}) than anisotropy ({anisotropy}).'
                )
            if np.sum(shape%np.array(scaling_factor))!=0:
                self.image_data = da.pad(self.image_data,pad_width=((0,0),(0,zpad),(0,ypad),(0,xpad)),mode='constant',constant_values=0)
    
    def save_spot_channels(self,index: int) -> None:
        """Save the spot channels of the image data in OME Zarr format.
        Args:
            index (int): Number of spot channels to save.
        """
        path_to_save = Path.cwd()/f'{self.image_name}-spotchannel.ome.zarr'
        omezarrwriter = OmeZarrWriter(path_to_save)
        if self.image_data.ndim==4:
            omezarrwriter.write_image(image_data=self.image_data[:index,:,:,:],
                                      image_name=self.image_name,
                                      physical_pixel_sizes=self.pixel_size,
                                      channel_colors=None,channel_names=None,
                                      dimension_order='CZYX')
        else:
            omezarrwriter.write_image(image_data=self.image_data[:index,:,:],
                                      image_name=self.image_name,
                                      physical_pixel_sizes=self.pixel_size,
                                      channel_colors=None,channel_names=None,
                                      dimension_order='CYX')
        print(f'Image has been saved at this location: {str(path_to_save)}',flush=True)
            
    def save_dapi(self,index: int) -> None:        
        """Save the DAPI channel of the image data in OME Zarr format.
        Args:
            index (int): Index of DAPI channel to save.
        """
        path_to_save = Path.cwd()/f'{self.image_name}-dapi.ome.zarr'
        omezarrwriter = OmeZarrWriter(path_to_save)
        if self.image_data.ndim==4:
            omezarrwriter.write_image(image_data=self.image_data[index,:,:,:],
                                      image_name=self.image_name,
                                      physical_pixel_sizes=self.pixel_size,
                                      channel_colors=None,channel_names=None,
                                      dimension_order='ZYX')
        else:
            omezarrwriter.write_image(image_data=self.image_data[index,:,:],
                                      image_name=self.image_name,
                                      physical_pixel_sizes=self.pixel_size,
                                      channel_colors=None,channel_names=None,
                                      dimension_order='YX')
        print(f'Image has been saved at this location: {str(path_to_save)}',flush=True)
            
    def save_brightfield(self,index: int) -> None:
        """Save the brightfield channel of the image data in TIFF format.
        Args:
            index (int): Index of brightfield channel to save.
        """
        if index > self.image_data.shape[0]-1:
            raise ValueError(f'Provided brightfield channel index ({index}) is larger than the last channel index ({self.image_data.shape[0]-1}).')
        path_to_save = Path.cwd()/f'{self.image_name}-brightfield.tif'
        
        if self.image_data.ndim==4:
            imsave(path_to_save,self.image_data[index,:,:,:].compute(),
                   plugin='tifffile')
        else:
            imsave(path_to_save,self.image_data[index,:,:].compute(),
                   plugin='tifffile')
        print(f'Image has been saved at this location: {str(path_to_save)}',flush=True)
    

if __name__ == "__main__":
    # def parse arguments
    def validate_path(file_path):
        if not Path(file_path).exists():
            raise argparse.ArgumentTypeError(f"File not found: {file_path}")
        return file_path

    def parse_arguments():
        parser = argparse.ArgumentParser(description="Split image into different channels and save them as OME Zarr and Tiff.")
        parser.add_argument("-im","--image_name",type=str,required=True,
                            help='Image name.')
        parser.add_argument("-f","--image_file", type=validate_path,required=True,
                            help='Path to image')
        parser.add_argument("--scene", type=int,required=True,
                            help='Index of scene to process')
        parser.add_argument("--spot_channels", type=int,required=True,
                            help='Number of spot channels')
        parser.add_argument("--dapi_channel", type=int,required=True,
                            help='Index of DAPI channel')
        parser.add_argument("--brightfield_channel", type=int,required=True,
                            help='Index of brightfield channel')
        parser.add_argument("--save_brightfield", action='store_true',
                            help='Option for saving brightfield image.')
        parser.add_argument("--scale_factor_xy", type=int,required=True,
                            help='Factor to downscale DAPI in xy image before segmentation.')
        parser.add_argument("--scale_factor_z", type=int,required=True,
                            help='Factor to downscale DAPI in z image before segmentation.')
        parser.add_argument("--anisotropy", type=float,required=True,
                            help='Image anisotropy in z.')
        parser.add_argument("--z_project", action='store_true',
                            help='Option to do a z projection on the image.')
        args = parser.parse_args()
        if args.anisotropy<=0:
            parser.error("--anisotropy must be an integer larger than 0.")
        return args
   
    args = parse_arguments()

    split_image = SplitImage(args.image_name,args.image_file,args.scene)
    split_image.reshape_image(args.z_project,
                              (args.scale_factor_z,args.scale_factor_xy,args.scale_factor_xy),
                              args.anisotropy)
    split_image.save_spot_channels(args.spot_channels)
    split_image.save_dapi(args.dapi_channel)
    if args.save_brightfield:
        split_image.save_brightfield(args.brightfield_channel)
