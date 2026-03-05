import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from statsmodels.distributions.empirical_distribution import ECDF

class FilterSpots:
    """Class to calculate a filtering threshold and filter the spots accordingly
    Attributes:
        spots_file (list): List of spot table files
        do_global_normalization (bool): Parameter to decide to globally normalise
        do_global_filtering (bool): Parameter to decide to globally filter
    Methods:
        process(self) -> None:
            Normalise and filter spots. Save filtered spot paths and image name as table.
        filter_spots(self,df: pd.DataFrame,kneepoint: np.ndarray,channels: list[str]) -> pd.DataFrame:
            Filter spots
        calculate_kneepoint(self,df: pd.DataFrame,channels: list[str]) -> np.ndarray:
            Calculate the knee point for each channel to use as threshold for filtering
        calculate_normalization_threshold(self,df: pd.DataFrame,channels: list[str]) -> pd.DataFrame:
            Calculate a threshold to normalise spot intensity
        normalize_spots(self,df: pd.DataFrame,threshold: pd.DataFrame,channels: list[str]) -> pd.DataFrame:
            Normalise spot intensities per channel
    """
    def __init__(self,
                 spots_file: list[str],
                 kneepoint_factor:float,
                 do_global_normalization:bool,
                 do_global_filtering:bool):
        self.spots_files = spots_file
        self.kneepoint_multiplication_factor = kneepoint_factor
        self.do_global_normalization = do_global_normalization
        self.do_global_filtering = do_global_filtering

    def process(self) -> None:
        """Normalise and filter spots. Save filtered spot paths and image name as table."""
        df_list = [pd.read_csv(file) for file in self.spots_files]
        if self.do_global_normalization:
            if len(df_list)==1:
                df_global = df_list[0]
            else:
                df_global = pd.concat(df_list).reset_index(drop=True)
            channels = df_global.columns[df_global.columns.str.contains(r'^Channel\d+$',regex=True)].tolist()
            normalization_threshold = self.calculate_normalization_threshold(df_global,channels)
            df_list = [self.normalize_spots(df,normalization_threshold,channels) for df in df_list]
        else:
            channels = [df.columns[df.columns.str.contains(r'^Channel\d+$',regex=True)].tolist() for df in df_list]
            normalization_threshold = [self.calculate_normalization_threshold(df,channel) for df,channel in zip(df_list,channels)]
            df_list = [self.normalize_spots(df,threshold,channel) for df,threshold,channel in zip(df_list,normalization_threshold,channels)]
        if self.do_global_filtering:
            if len(df_list)==1:
                df_global = df_list[0]
            else:
                df_global = pd.concat(df_list).reset_index(drop=True)
            channels = df_global.columns[df_global.columns.str.contains(r'^Channel\d+$',regex=True)].tolist()
            kneepoints = self.calculate_kneepoint(df_global,channels)
            df_list = [self.filter_spots(df,kneepoints,channels) for df in df_list]
        else:
            channels = [df.columns[df.columns.str.contains(r'^Channel\d+$',regex=True)].tolist() for df in df_list]
            kneepoints = [self.calculate_kneepoint(df,channel) for df,channel in zip(df_list,channels)]
            df_list = [self.filter_spots(df,kneepoint,channel) for df,kneepoint,channel in zip(df_list,kneepoints,channels)]
        names = []
        paths = []
        for df in df_list:
            image_name = df['name'].unique()[0]
            names.append(image_name)
            path_to_save = Path.cwd()/f'{image_name}_filtered_spots.csv'
            paths.append(path_to_save)
            df.to_csv(path_to_save,index=False)
        pd.DataFrame({'name':names,'path':paths}).to_csv(Path.cwd()/'Filtered_Spots.csv',index=False)
        if isinstance(normalization_threshold,list):
            for t,name in zip(normalization_threshold,names):
                t.insert(0,column='name',value=name)
            normalization_threshold = pd.concat(normalization_threshold).reset_index(drop=True)
        if isinstance(kneepoints,list):
            kneepoints_df = []
            for k,name,c in zip(kneepoints,names,channels):
                k = pd.DataFrame({'channel':c,'kneepoint':k})
                k.insert(0,column='name',value=name)
                kneepoints_df.append(k)
            kneepoints = pd.concat(kneepoints_df).reset_index(drop=True)
            del kneepoints_df
        else:
            kneepoints = pd.DataFrame({'channel':channels,'kneepoint':kneepoints})
        path_to_save = Path.cwd()/f'spot_normalization_threshold.csv'
        if 'name' in normalization_threshold.columns and 'name' in kneepoints.columns:
            df = pd.merge(normalization_threshold,kneepoints,on=['name','channel'])
        elif 'name' in normalization_threshold.columns and 'name' not in kneepoints.columns:
             df = pd.merge(normalization_threshold,kneepoints,on='channel')
        elif 'name' not in normalization_threshold.columns and 'name' in kneepoints.columns:
            df = pd.merge(normalization_threshold,kneepoints,on='channel',how='right')
            df = df[['name','channel','intensity_p998','kneepoint']]
        else:
            df = pd.merge(normalization_threshold,kneepoints,on='channel')
        if 'name' in df.columns:
            df.sort_values(by='name')
            df.to_csv(path_to_save,index=False)
        else:
            df.to_csv(path_to_save,index=False)

    def filter_spots(self,
                     df: pd.DataFrame,
                     kneepoint: np.ndarray,
                     channels: list[str]) -> pd.DataFrame:
        """Filter spots.
        Args:
            df (pd.DataFrame): Table of spots
            kneepoint (np.ndarray): Threshold values for each channel
            channels (list): list of channel names
        Returns:
            pd.DataFrame: Table of filtered spots
        """
        kneepoint = pd.Series(kneepoint, index=channels,name='threshold')
        df = df.merge(kneepoint,left_on='channel',right_index=True)
        df = df[df['intensity']>df['threshold']].reset_index(drop=True)
        df['true_single'] = df['intensity']>self.kneepoint_multiplication_factor*df['threshold']
        df['intensity'] = df['intensity'] - df['threshold']
        # Subtract thresholds from the intensity columns in place
        for ch in channels:
            df[ch] = df[ch] - kneepoint[ch]
        df.drop(columns='threshold',inplace=True)
        return df


    def calculate_kneepoint(self,
                            df: pd.DataFrame,
                            channels: list[str]) -> np.ndarray:
        """Calculate the knee point for each channel to use as threshold for filtering
        Args:
            df (pd.DataFrame): Table of spots
            channels (list): List of channel names
        Returns:
            np.ndarray: Knee point values for each channel 
        """
        kneepoints = []
        for channel in channels:
            channel_values = np.sort(df[channel].values)
            ecdf = ECDF(channel_values)
            ecdf_values = ecdf(channel_values)
            norm_value = (channel_values-channel_values.min())/(channel_values.max()-channel_values.min())
            norm_ecdf_values = (ecdf_values-ecdf_values.min())/(ecdf_values.max()-ecdf_values.min())
            dist_to_diag = norm_ecdf_values-norm_value
            knee_index = np.argmax(dist_to_diag)
            kneepoints.append(channel_values[knee_index])
        return np.array(kneepoints)

    def calculate_normalization_threshold(self,
                                          df: pd.DataFrame,
                                          channels: list[str]) -> pd.DataFrame:
        """Calculate a threshold to normalise spot intensity
        Args:
            df (pd.DataFrame): Table of spots
            channels (list): List of channel names
        Returns:
            pd.DataFrame: Table of intensity threshold for normalisation
        """
        intensity_p998 = [np.percentile(df[channel].values,99.8) for channel in channels]
        return pd.DataFrame({'channel':channels,'intensity_p998':intensity_p998})
    
    def normalize_spots(self,
                        df: pd.DataFrame,
                        threshold: pd.DataFrame,
                        channels: list[str]) -> pd.DataFrame:
        """Normalise spot intensities per channel
        Args:
            df (pd.DataFrame): Table of spots
            threshold (pd.DataFrame): Table of channel specific normalisation values
            channel (list): List of channel names
        Returns:
            pd.DataFrame: Table of normalised spots
        """
        for channel in channels:
            df[channel] /= threshold[threshold['channel']==channel]['intensity_p998'].values[0]
        df = pd.merge(df,threshold,on='channel',how='left')
        df['intensity'] /= df['intensity_p998']
        df.drop(columns='intensity_p998',inplace=True)
        return df


if __name__ == "__main__":
    # def parse arguments
    def validate_path(file_path):
        if not Path(file_path).exists():
            raise argparse.ArgumentTypeError(f"File not found: {file_path}")
        return file_path

    def parse_arguments():
        parser = argparse.ArgumentParser(description="Filter spots after detection")
        parser.add_argument("-s","--spots", type=validate_path,required=True,nargs='+',
                            help='Path to spot csv.')
        parser.add_argument("--global_normalisation", action='store_true',
                            help='Whether to perform global normalization across all images.')
        parser.add_argument("--global_filtering", action='store_true',
                            help='Whether to perform global filtering across all images.')
        parser.add_argument("--kneepoint_factor", type=float,required=True,default=1.6,
                            help='Multiplikation factor for kneepoint threshold.')
        args = parser.parse_args()
        return args

    args = parse_arguments()
    FilterSpots(args.spots,args.kneepoint_factor,args.global_normalisation,args.global_filtering).process()