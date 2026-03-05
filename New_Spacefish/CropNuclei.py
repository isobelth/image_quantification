from bioio import BioImage
from bioio.writers import OmeTiffWriter
from bioio_base.types import PhysicalPixelSizes
import argparse
import pandas as pd
from skimage.measure import regionprops,perimeter
import skimage
from skimage.io import imsave
from pathlib import Path
import dask.bag as db
import numpy as np


class CropNuclei:
    """Class to crop nuclei based on segmentation mask.
    Attributes:
        image_name (str): Name of the image .
        spotchannels (dask.array.Array): Dask array containing the spotchannel data.
        dapichannel (dask.array.Array): Dask array containing the DAPI data.
        pixel_size (bioio_base.types.PhysicalPixelSizes): Physical pixel sizes of the image
        seg_mask (dask.array.Array): Dask array containing the segmentation data.
        image_shape (tuple): Shape of the segmentation mask
        buffer (int): Buffer in z for cropping nucleus before max z projection
    
    Methods:
        filter_out_nuclei(self,filter_convexity: bool,convexity_threshold: float,
                          additional_cleanup: bool,pixel_size: np.ndarray,nucleus_area:float) -> None:
            Filter out nuclei based on convexity and additional cleanup steps.
        get_lower_and_upper_bounds(values: list|np.ndarray) -> tuple[float,float]:
            Get lower and upper bounds based on IQR.
        get_convexity(prop: skimage.measure._regionprops.RegionProperties) -> float:
            Calculate nucleus convexity
        process(self) -> None:
            Process the image by cropping it into individual nuclei and saving statistics.
        crop_image(crop_input: tuple, image_shape: tuple, seg_mask: np.ndarray,
                   dapichannel: np.ndarray, spotchannels: np.ndarray, buffer: int,
                   image_name: str) -> pd.DataFrame | None:
            Crop the image based on the bounding box and nucleus id. Saves the cropped image as a TIFF and OME-Zarr file.   
    """


    def __init__(self,
                 image_name: str,
                 spotchannels: str,
                 dapichannel: str,
                 seg_mask: str,
                 buffer: int,
                 pixel_size: np.ndarray=np.array([1,1,1]),) -> None:
        self.image_name = image_name
        self.spotchannels = BioImage(spotchannels).dask_data.squeeze().compute()
        self.dapichannel = BioImage(dapichannel).dask_data.squeeze().compute()
        self.pixel_size = pixel_size
        self.seg_mask = BioImage(seg_mask).dask_data.squeeze().compute()
        self.image_shape = self.seg_mask.shape
        self.buffer = buffer

    def filter_out_nuclei(self,
                          filter_convexity: bool=False,
                          convexity_threshold: float=0.7,
                          additional_cleanup: bool=False,
                          nucleus_area: float=20.0,
                          intensity_threshold: float=4.0) -> None:
        """Filter out nuclei based on convexity and additional cleanup steps.
        Args:
            filter_convexity (bool): Whether to filter for convexity or not.
            convexity_threshold (float): Threshold to filter convexity. Should be between 0 and 1
            additional_cleanup (bool): Whether to perform additional cleanup steps.
        """
        if filter_convexity and self.seg_mask.ndim==2:
            rp = regionprops(self.seg_mask,spacing=tuple(self.pixel_size[1:]))
            ids_to_remove = [prop.label for prop in rp if CropNuclei.get_convexity(prop,self.pixel_size[-1])<convexity_threshold]
            self.seg_mask = np.where(np.isin(self.seg_mask,ids_to_remove),0,self.seg_mask)
        if additional_cleanup:
            if self.seg_mask.ndim==3:
                df_list = []
                for z in range(self.seg_mask.shape[0]):
                    if self.seg_mask[z].max()==0:
                        continue
                    rp = regionprops(self.seg_mask[z], spacing=tuple(self.pixel_size[1:]),intensity_image=self.dapichannel[z])
                    df = pd.DataFrame({'label': [prop.label for prop in rp],
                                       'area': [prop.area for prop in rp],
                                       'convexity': [CropNuclei.get_convexity(prop,self.pixel_size[-1]) for prop in rp],
                                       'mean_intensity': [prop.mean_intensity for prop in rp]})
                    df['z'] = z
                    df_list.append(df)
                    del df
                df_list = pd.concat(df_list)
                label_ids, label_count = np.unique(df_list['label'], return_counts=True)
                ids_to_remove = list(label_ids[label_count==1].astype('int'))
                self.seg_mask = np.where(np.isin(self.seg_mask,ids_to_remove),0,self.seg_mask)
                label_count = pd.DataFrame({'label': label_ids, '3D': label_count!=1})
                df_list = df_list.merge(label_count, on='label', how='left')
                df_list = df_list[df_list['3D']]
                df_list = df_list[(df_list['area']>nucleus_area) |  (df_list['convexity']<convexity_threshold) | (df_list['mean_intensity'] < intensity_threshold)]
                new_mask = []
                for z in range(self.seg_mask.shape[0]):
                    mask = self.seg_mask[z]
                    ids_to_remove = df_list[df_list['z']==z]
                    if not ids_to_remove.empty:
                        ids_to_remove = ids_to_remove['label'].values.astype('int')
                        mask = np.where(np.isin(mask,ids_to_remove),0,mask)
                    new_mask.append(mask)
                    del mask
                self.seg_mask = np.stack(new_mask,axis=0)
                del new_mask, df_list
            else:
                rp = regionprops(self.seg_mask, spacing=tuple(self.pixel_size[1:]),intensity_image=self.dapichannel)
                df = pd.DataFrame({'label': [prop.label for prop in rp],
                                   'area': [prop.area for prop in rp],
                                   'convexity': [CropNuclei.get_convexity(prop,self.pixel_size[-1]) for prop in rp],
                                   'mean_intensity': [prop.mean_intensity for prop in rp] })
                ids_to_remove = df[(df['area']>nucleus_area) |  (df['convexity']<convexity_threshold) | (df['mean_intensity'] < intensity_threshold)]['label'].values.astype('int')
                self.seg_mask = np.where(np.isin(self.seg_mask,ids_to_remove),0,self.seg_mask)
                del df

    @staticmethod
    def get_convexity(prop: skimage.measure._regionprops.RegionProperties,pixel_size: float) -> float:
        """Calculate nucleus convexity
        Args:
            prop (skimage.measure._regionprops.RegionProperties): Nucleus region properties
        Returns:
            float: Nucleus convexity
        """
        perimeter_value = prop.perimeter
        convex_hull = prop.convex_image
        convex_perimeter = perimeter(convex_hull)*pixel_size
        return convex_perimeter / perimeter_value if perimeter_value > 0 else 0
    
    def process(self) -> None:
        """Process the image by cropping it into individual nuclei and saving statistics."""
        path_to_save = Path.cwd()/f'{self.image_name}-seg_mask.tif'
        imsave(path_to_save,self.seg_mask,check_contrast=False)
        print(f'Segmentation mask has been saved at this location: {str(path_to_save)}',flush=True)
        rp = regionprops(self.seg_mask)
        crop_input = [(prop.bbox,prop.label) for prop in rp]
        crop_input = db.from_sequence(crop_input,npartitions=min(127,len(crop_input)-1))
        df = crop_input.map(CropNuclei.crop_image,self.image_shape,self.seg_mask,self.dapichannel,
                            self.spotchannels,self.buffer,self.image_name).compute(scheduler='threads')
        df = [x for x in df if x is not None]
        if not df:
            raise ValueError("No valid crops to process.")
        df = pd.concat(df,axis=0,ignore_index=True)
        path_to_save = Path.cwd()/f'{self.image_name}-NucleiLocation.csv'
        df.to_csv(path_to_save,index=False)
        print(f'Location of nuclei are saved at {str(path_to_save)}',flush=True)
    
    @staticmethod
    def crop_image(crop_input: tuple,
                   image_shape: tuple,
                   seg_mask: np.ndarray,
                   dapichannel: np.ndarray,
                   spotchannels: np.ndarray,
                   buffer: int,
                   image_name: str) -> pd.DataFrame | None:
        """Crop the image based on the bounding box and nucleus id. Saves the cropped image as a TIFF and OME-Zarr file.
        Args:
            crop_input (tuple): Bounding box coordinates and ID of the nucleus to crop
            image_shape (tuple): Shape of the image
            seg_mask (np.ndarray): Numpy array containing the segmentation mask
            dapichannel (np.ndarray): Numpy array containing the DAPI channel
            spotchannels np.ndarray): Numpy array containing the spot channels
            buffer (int): Buffer in z for cropping nucleus before max z projection
            image_name (str): Name of the image
        Returns:
            pd.DataFrame: Nucleus global coordinates.
        """
        bbox,nucleus_id = crop_input
        if len(bbox)==6:
            z_low,y_low,x_low,z_upper,y_upper,x_upper = bbox
            y_low = max(y_low-25,0)
            x_low = max(x_low-25,0)
            y_upper = min(y_upper+25,image_shape[1])
            x_upper = min(x_upper+25,image_shape[2])
            if z_upper <= z_low or y_upper <= y_low or x_upper <= x_low:
                print(f"Skipping nucleus {nucleus_id}: empty crop (bbox={bbox})")
                return None
            mask = seg_mask[z_low:z_upper,y_low:y_upper,x_low:x_upper]
            mask = np.where(mask!=nucleus_id,0,mask)
            mask = mask.max(axis=0)
            dapi = dapichannel[z_low:z_upper,y_low:y_upper,x_low:x_upper].max(axis=0)
            z_upper = min(z_upper+buffer,image_shape[0])
            img = spotchannels[:,z_low:z_upper,y_low:y_upper,x_low:x_upper]
            if np.any(np.array(img.shape)<1):
                print(f'Skipping nucleus {nucleus_id}: corrupted crop (shape={img.shape})')
                return None
            img = img.max(axis=1)
            img = np.concatenate([img, dapi[None,:], mask[None,:]], axis=0)
            mask = seg_mask[z_low:z_upper,y_low:y_upper,x_low:x_upper]
            mask = np.where(mask!=nucleus_id,0,mask)
            dapi = dapichannel[z_low:z_upper,y_low:y_upper,x_low:x_upper]
            spot_channel = spotchannels[:,z_low:z_upper,y_low:y_upper,x_low:x_upper]
            spot_channel = np.concatenate([spot_channel, dapi[None,:], mask[None,:]], axis=0)
            path_to_save = Path.cwd()/f'{image_name}-nucleus{nucleus_id:04d}_cropped_3D.ome.tif'
            OmeTiffWriter().save(spot_channel,path_to_save)
        else:
            y_low,x_low,y_upper,x_upper = bbox
            y_low = max(y_low-25,0)
            x_low = max(x_low-25,0)
            y_upper = min(y_upper+25,image_shape[0])
            x_upper = min(x_upper+25,image_shape[1])
            if y_upper <= y_low or x_upper <= x_low:
                print(f"Skipping nucleus {nucleus_id}: empty crop (bbox={bbox})")
                return None
            mask = seg_mask[y_low:y_upper,x_low:x_upper]
            mask = np.where(mask!=nucleus_id,0,mask)
            dapi = dapichannel[y_low:y_upper,x_low:x_upper]
            img = spotchannels[:,y_low:y_upper,x_low:x_upper]
            img = np.concatenate([img, dapi[None,:], mask[None,:]], axis=0)
        path_to_save = Path.cwd()/f'{image_name}-nucleus{nucleus_id:04d}_cropped.tif'
        imsave(path_to_save,img,check_contrast=False)
        print(f'Image has been saved at this location: {str(path_to_save)}',flush=True)
        return pd.DataFrame({'image':[image_name],'nucleus':[nucleus_id],
                                'ystart':[y_low],
                                'xstart':[x_low]})

    
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
        parser.add_argument("-s","--spotchannel", type=validate_path,required=True,
                            help='Path to spotchannel image')
        parser.add_argument("-d","--dapi", type=validate_path,required=True,
                            help='Path to dapi image')
        parser.add_argument("-m","--mask", type=validate_path,required=True,
                            help='Path to segmentation mask')
        parser.add_argument("-b","--buffer", type=int,required=True,
                            help='Buffer in z if to processing large z stacks.')
        parser.add_argument("--additional_cleanup", action='store_true',
                            help='Whether additional cleanup should be performed.')
        parser.add_argument("-ct","--convexity_threshold", type=float,required=False,
                            help='Threshold for filtering nuclei for convexity.')
        parser.add_argument("--filter_convexity", action='store_true',
                            help='Whether to filter for convexity in this step or not.')
        parser.add_argument("--res_xy", type=float,required=False, default=1,
                            help='Resolution in xy in microns.')
        parser.add_argument("--res_z", type=float,required=False, default=1,
                            help='Resolution in z in microns.')
        parser.add_argument("--nucleus_area", type=float,required=True,default=20,
                            help='Threshold for maximal nucleus area in 2D.')
        parser.add_argument("--intensity_threshold", type=float,required=True,default=4,
                            help='Intensity threshold to filter out bad segmentation masks.')
        return parser.parse_args()
   
    args = parse_arguments()

    cropnuclei = CropNuclei(args.image_name,args.spotchannel,args.dapi,args.mask,args.buffer,pixel_size=np.array([args.res_z,args.res_xy,args.res_xy]))
    if args.filter_convexity or args.additional_cleanup:
        cropnuclei.filter_out_nuclei(filter_convexity=args.filter_convexity,
                                     convexity_threshold=args.convexity_threshold if args.convexity_threshold else 0.7,
                                     additional_cleanup=args.additional_cleanup,
                                     nucleus_area = args.nucleus_area,
                                     intensity_threshold=args.intensity_threshold)
    cropnuclei.process()
    
