from bioio import BioImage
from spotiflow.model import Spotiflow
import spotiflow
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import re
from skimage.transform import resize
from scipy.spatial import KDTree

class SpotDetector:
    """Class to detect spots in nuclei
    Attributes:
        image_name (str): Image name
        image_path (list): List of image paths
    Methods:
        extract_nucleus_id(self,file: str) -> int:
            Extract nucleus ID from file name
        detect(self,model: str,min_distance: int) -> None:
            Does spot detection on each nucleus and save spots as table
        detect_spots(self,file: str,spotiflow_model: spotiflow.model.spotiflow.Spotiflow,
                        min_distance: int) -> tuple[pd.DataFrame,int]:
            Detects spots in single nucleus
        remove_duplicates(self,df: pd.DataFrame,number_of_channels: int,min_distance: int) -> pd.DataFrame:
            Remove spots per channel that are detected more than once
        detect_spots_all_channels(self,img: np.ndarray,img_pad: np.ndarray,
                                  spotiflow_model: spotiflow.model.spotiflow.Spotiflow,
                                  min_distance: int,scale_factor: int,nucleus_id: int) -> pd.DataFrame|None:
            Detect spot in all channels and interpolate the intensity at this position
        interpolate(self,img: np.ndarray,points: pd.DataFrame, channel:int) -> np.ndarray:
            Interpolate the spot intensities over all channels at a specific spot location
        detect_spots_in_single_channel(self,img: np.ndarray,model: spotiflow.model.spotiflow.Spotiflow,
                                       min_distance: int,scale_factor: int) -> pd.DataFrame|None:
            Detect spots in a single channel
    """

    def __init__(self,
                 image_name: str,
                 image_path: list[str]) -> None:
        self.image_name = image_name
        self.image_path = image_path
    
    def extract_nucleus_id(self,file: str) -> int:
        """Extract nucleus ID from file name
        Args:
            file (str): Path to image
        Returns:
            int: Nucleus ID
        """
        nucleus_id = Path(file).stem.split('-')[1]
        nucleus_id = re.search(r'\d+',nucleus_id).group()
        return int(nucleus_id)
    
    def detect(self,
               model: str,
               min_distance: int) -> None:
        """Does spot detection on each nucleus and save spots as table
        Args:
            model (str): Spot detection model to use
            min_distance (int): Minimum distance between to spots to detect them seperately
        """
        spotiflow_model = Spotiflow.from_pretrained(model)
        df = []
        for file in self.image_path:
            df_nucleus,id = self.detect_spots(file,spotiflow_model,min_distance)
            if isinstance(df_nucleus,pd.DataFrame):
                df.append(df_nucleus)
        if len(df)>0:
            if len(df)>1:
                df = pd.concat(df,axis=0,ignore_index=True)
            else:
                df = df[0]
            path_to_save = Path.cwd()/f'{self.image_name}-nuclei{id}-spots.csv'
            df.to_csv(path_to_save,index=False)
            print(f'Spots are saved at {path_to_save}.')
        else:
            print(f'No spots where detected.')

    def detect_spots(self,
                     file: str,
                     spotiflow_model: spotiflow.model.spotiflow.Spotiflow,
                     min_distance: int) -> tuple[pd.DataFrame,int]:
        """Detects spots in single nucleus
        Args: 
            file (str): Path to image data
            spotiflow_model (spotiflow.model.spotiflow.Spotiflow): Loaded spotiflow model
            min_distance (int): Minimum distance between to spots to detect them seperately
        Returns:
            pd.DataFrame: Table of detected spots
            int: Nucleus ID
        """
        img = BioImage(file).dask_data.squeeze().compute()
        # extract nucleus_id
        nucleus_id = self.extract_nucleus_id(file)
        # remove out of nucleus signal
        nucleus_mask = img[-1] > 0
        mask = ~nucleus_mask
        img = img[:-2]
        # mask = np.stack([mask]*img.shape[0],axis=0)
        img[:,mask]=0
        scale_factor = 2
        img1 = resize(img,(img.shape[0],img.shape[1]//scale_factor,img.shape[2]//scale_factor),order=0,anti_aliasing=True)
        img_pad = np.pad(img,((0,0),(2,2),(2,2)),mode='reflect')
        # first do spot detection on the original image
        df_nucleus_original = self.detect_spots_all_channels(img,img_pad,spotiflow_model,min_distance,1,nucleus_id,nucleus_mask)
        df_nucleus_resized = self.detect_spots_all_channels(img1,img_pad,spotiflow_model,min_distance,scale_factor,nucleus_id,nucleus_mask)
        if isinstance(df_nucleus_original,pd.DataFrame):
            if isinstance(df_nucleus_resized,pd.DataFrame):
                df_nucleus = pd.concat([df_nucleus_original,df_nucleus_resized],axis=0,ignore_index=True)
                df_nucleus = self.remove_duplicates(df_nucleus,img.shape[0],min_distance)
                df_nucleus.insert(0,column='nucleus',value=nucleus_id)
                df_nucleus.insert(0,column='name',value=self.image_name)
            else:
                df_nucleus = df_nucleus_original
        else:
            if isinstance(df_nucleus_resized,pd.DataFrame):
                df_nucleus = df_nucleus_resized
            else:
                df_nucleus = []
                print(f'No spots where detected in neither channel of nucleus {nucleus_id} of image {self.image_name}.')
        return df_nucleus,nucleus_id
    
    def remove_duplicates(self,
                          df: pd.DataFrame,
                          number_of_channels: int,
                          min_distance: int) -> pd.DataFrame:
        """Remove spots per channel that are detected more than once
        Args:
            df (pd.DataFrame): Table of detected spots
            number_of_channels (int): Number of spot channels
            min_distance (int): Minimum distance between to spots to detect them seperately
        Returns:
            pd.DataFrame: Table of filtered spots
        """
        channels = np.unique(df['channel'])
        intensity_columns = ['intensity'] + [f'Channel{j+1}' for j in range(number_of_channels)]
        df_all = []
        for channel in channels:
            df_channel = df[df['channel']==channel].copy().reset_index(drop=True)
            coords = df_channel[['y','x']].values
            tree = KDTree(coords)
            duplicates = tree.query_ball_point(coords,r=min_distance,workers=-1)
            duplicates = [tuple(sorted(d)) for d in duplicates]
            duplicates = list(set(duplicates))   
            new_coords = [df_channel.loc[list(d)][['y','x']].mean(axis=0) for d in duplicates]
            df_coord = pd.DataFrame(np.array(new_coords),columns=['y','x'])
            new_intensity = [df_channel.loc[list(d)][[ic for ic in intensity_columns]].max(axis=0) for d in duplicates]
            df_intensity = pd.DataFrame(np.array(new_intensity),columns=intensity_columns)
            df_channel = pd.concat([df_coord,df_intensity],axis=1)
            df_channel.insert(0,column='channel',value=channel)
            df_all.append(df_channel)
        return pd.concat(df_all,axis=0,ignore_index=True)            
        
    def detect_spots_all_channels(self,
                                  img: np.ndarray,
                                  img_pad: np.ndarray,
                                  spotiflow_model: spotiflow.model.spotiflow.Spotiflow,
                                  min_distance: int,
                                  scale_factor: int,
                                  nucleus_id: int,
                                  nucleus_mask: np.ndarray) -> pd.DataFrame|None:
        """Detect spot in all channels and interpolate the intensity at this position
        Args:
            img (np.ndarray): Image data for spot detection
            img_pad (np.ndarray): Padded image data for intensity interpolation
            spotiflow_model (spotiflow.model.spotiflow.Spotiflow): Loaded spotiflow model
            min_distance (int): Minimum distance between to spots to detect them seperately
            scale_factor (int): Scale factor for second spot detection
            nucleus_id (int): Nucleus ID
        Returns:
            pd.DataFrame: Table of detected spots or None if no spots are detected
        """
        df_nucleus = []
        for i,channel in enumerate(img):
            print(f'Current channel: {i}')
            df_channel = self.detect_spots_in_single_channel(channel,spotiflow_model,min_distance,scale_factor)
            if isinstance(df_channel,pd.DataFrame):
                df_channel = self.filter_points_to_nucleus_mask(df_channel,nucleus_mask)
                if df_channel is None or df_channel.empty:
                    print(f'No in-mask spots remained in channel {i+1} of nucleus {nucleus_id}.')
                    continue
                df_channel.insert(0,column='channel',value=f'Channel{i+1}')
                spots = df_channel[['y','x']].values
                intensity = self.interpolate(img_pad,spots,i)
                columns = ['intensity']+[f'Channel{j+1}' for j in range(img.shape[0])]
                index = [i]+[j for j in range(img.shape[0])]
                df_intensity = pd.DataFrame({column:intensity[:,j] for column,j in zip(columns,index)})
                df_nucleus.append(pd.concat([df_channel,df_intensity],axis=1))
            else:
                print(f'No spots where detected in channel {i+1} of nucleus {nucleus_id}.')
        if len(df_nucleus)>0:
            if len(df_nucleus)>1:
                df_nucleus = pd.concat(df_nucleus,axis=0,ignore_index=True)
            else:
                df_nucleus = df_nucleus[0]
            df_nucleus.insert(0,column='nucleus',value=nucleus_id)
            df_nucleus.insert(0,column='name',value=self.image_name)
            return df_nucleus
        else:
            print(f'No spots where detected in neither channel of nucleus {nucleus_id} of image {self.image_name}.')
            return None

    def filter_points_to_nucleus_mask(self,
                                      points_df: pd.DataFrame,
                                      nucleus_mask: np.ndarray) -> pd.DataFrame:
        """Keep only spots whose rounded coordinates lie inside the nucleus mask."""
        if points_df is None or points_df.empty:
            return points_df
        if not {'y','x'}.issubset(points_df.columns):
            return points_df
        y_idx = np.rint(points_df['y'].values).astype(int)
        x_idx = np.rint(points_df['x'].values).astype(int)
        in_bounds = (
            (y_idx >= 0)
            & (x_idx >= 0)
            & (y_idx < nucleus_mask.shape[0])
            & (x_idx < nucleus_mask.shape[1])
        )
        keep = np.zeros(len(points_df),dtype=bool)
        keep[in_bounds] = nucleus_mask[y_idx[in_bounds],x_idx[in_bounds]]
        return points_df.loc[keep].reset_index(drop=True)

    def interpolate(self,
                    img: np.ndarray,
                    points: pd.DataFrame,
                    channel:int) -> np.ndarray:
        """Interpolate the spot intensities over all channels at a specific spot location
        Args:
            img (np.ndarray): Image data
            points (pd.DataFrame): Table of spot coordinates
            channel (int): Channel index (zero-based)
        Returns:
            np.ndarray: spot intensity vectors
        """
        intensity_vec = []
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),  (0, 0),  (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]
        for point in points:
            point = np.round(point).astype('int') + 2
            max_mean = []
            for dy, dx in offsets:
                cy, cx = point[0] + dy, point[1] + dx
                region = img[channel, cy-1:cy+2, cx-1:cx+2]
                if region.shape == (3, 3):  # Check bounds
                    max_mean.append((cy,cx,region.mean()))
            max_mean = sorted(max_mean,key=lambda x: x[2],reverse=True)
            print(max_mean)
            y = max_mean[0][0]
            x = max_mean[0][1]
            print(y,x)
            intensity_vec.append(img[:, y-1:y+2, x-1:x+2].mean(axis=(1,2)))
        return np.stack(intensity_vec, axis=0)


    def detect_spots_in_single_channel(self,
                                       img: np.ndarray,
                                       model: spotiflow.model.spotiflow.Spotiflow,
                                       min_distance: int,
                                       scale_factor: int) -> pd.DataFrame|None:
        """Detect spots in a single channel
        Args:
            img (np.ndarray): Image data for spot detection
            model (spotiflow.model.spotiflow.Spotiflow): Loaded spotiflow model
            min_distance (int): Minimum distance between to spots to detect them seperately
            scale_factor (int): Scale factor for second spot detection
        Returns:
            pd.DataFrame: Table of detected spots or None if no spots are detected
        """
        points, _ = model.predict(img,subpix=True,min_distance=min_distance)
        if points.shape[0]!=0:
            img_shape = np.array(img.shape)-1
            points = np.where(points>img_shape,img_shape,points)
            if points.shape[1]==2:
                return pd.DataFrame(points.astype('float16')*scale_factor,columns=['y','x'])
            else:
                return pd.DataFrame(points.astype('float16')*np.array([1,scale_factor,scale_factor]),columns=['z','y','x'])
        else:
            return None

if __name__ == "__main__":
    # def parse arguments
    def validate_path(file_path):
        if not Path(file_path).exists():
            raise argparse.ArgumentTypeError(f"File not found: {file_path}")
        return file_path

    def validate_model(model):
        valid_models = ('general','hybiss','synth_complex')
        if model not in valid_models:
            raise argparse.ArgumentTypeError(f"Invalid model selection: {model}. Must be one of {valid_models}.")
        return model

    def parse_arguments():
        parser = argparse.ArgumentParser(description="Detecting spots")
        parser.add_argument("-im","--image_name",type=str,required=True,
                            help='Image name.')
        parser.add_argument("-f","--image_path", type=validate_path,required=True,nargs='+',
                            help='Paths to images.')
        parser.add_argument("-m","--model", type=validate_model,required=True,
                            help='Spotiflow model')
        parser.add_argument("--min_distance", type=int,required=True,
                            help='Minimal distance between spots.')
        args = parser.parse_args()
        if args.min_distance < 0:
            parser.error("--min_distance must be a positive integer.")
        return args
    
   
    args = parse_arguments()

    SpotDetector(args.image_name,args.image_path).detect(args.model,args.min_distance)
