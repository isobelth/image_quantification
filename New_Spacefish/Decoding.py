from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import re
import networkx as nx
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
import argparse

class Codebook:
    """Class for loading the codebook.
    Attributes
        codebook (dict): Codebook with channel combination as keys and gene names as values
    Methods
        add_channel_info(self,channel_info: str) -> dict:
            Translate channel info file into dictionary
        add_codebook(self,codebook_path: str,channel_info: str,number_of_channels: int) -> dict:
            Creates a codebook out of two files
     """
    
    def __init__(self,
                 codebook_path: str,
                 channel_info: str,
                 number_of_channels: int) -> None:
        self.codebook = self.add_codebook(codebook_path=codebook_path,
                                          channel_info=channel_info,
                                          number_of_channels=number_of_channels)
        
    def add_channel_info(self,channel_info: str) -> dict:
        """Translate channel info file into dictionary
        Args:
            channel_info (str): Path to channel info file
        Returns:
            dict: Key are fluorophores and values are channel name and index
        """
        channel_info = pd.read_csv(channel_info,delimiter=' ',names=['channel','fluorophore'])
        channel_info['channel'] = [f'Channel{i+1}' for i in range(len(channel_info))]
        return {row['fluorophore']:[row['channel'],i] for i,row in channel_info.iterrows()}
    
    def add_codebook(self,
                     codebook_path: str,
                     channel_info: str,
                     number_of_channels: int) -> dict:
        """Creates a codebook out of two files
        Args:
            codebook_path (str): Path to codebook file
            channel_info (str): Path to channel info file
            number_of_channels (int): Number of spot channels
        Returns:
            dict: Codebook with channel combination as keys and gene names as values
        """
        channel_info = self.add_channel_info(channel_info=channel_info)
        codebook_df = pd.read_csv(codebook_path)
        columns = list(codebook_df.columns)
        codebook_dict = {}
        for i,row in codebook_df.iterrows():
            barcode = np.zeros(number_of_channels,dtype='int8')
            channel_list = []
            for fluorophore in [row[c] for c in columns[1:]]:
                if fluorophore in channel_info.keys():
                    channel,i = channel_info[fluorophore]
                    barcode[i] = 1
                    channel_list.append(channel)
            channel_list = sorted(channel_list)
            if len(channel_list)==1:
                channel_list = int(re.search(r'\d+',channel_list[0]).group())
            else:
                channel_list = [re.search(r'\d+',ch).group() for ch in channel_list]
                channel_list = int("".join(map(str,channel_list)))
            codebook_dict[channel_list] = row['gene']
        return codebook_dict
    
    def __str__(self):
        return f'Object of class Codebook. Class attribute is codebook (dictionary) with channel combination as keys and gene name as values.'
    

class Decoding:
    """Class to decode spots into barcodes
    Attributes:
        image_name (str): Image name
        spots (pd.DataFrame): Table of spots
        codebook (dict): Codebook with channel combination as keys and gene names as values
        channel (list): List of Channel names
        radius (int|float): Search radius for neighboring spots
        overlap_threshold (float): Threshold for channel overlap
        balance_threshold (float): Threshold for channel intensity balance
    Methods:
        add_codebook(self,codebook_path: str,channel_info: str,number_of_channels: int) -> None:
            Load codebook for decoding
        decode(self) -> None:
            Decode spots into barcodes for each nucleus
        get_gene_barcode(self,barcode: int) -> str:
            Returns gene name for given channel combination
        build_graph(self,df: pd.DataFrame) -> nx.Graph:
            Build graph of connected spots
        decode_single(self,df: pd.DataFrame,idx: int,nucleus: int) -> list:
            Decode single spots
        decode_multi(self,df: pd.DataFrame,G: nx.Graph,indices: set,nucleus: int) ->list:
            Decode group of connected spots
    """

    def __init__(self,
                 image_name: str,
                 spots: str,
                 noc: int,
                 radius: int|float,
                 overlap_threshold: float,
                 balance_threshold: float) -> None:
        self.image_name = image_name
        self.spots = pd.read_csv(spots)
        self.codebook = {}
        self.channels = [f'Channel{i+1}' for i in range(noc)]
        intensities = self.spots[['intensity'] + self.channels].values
        intensities = np.clip(intensities,a_min=0,a_max=None)
        self.spots[['intensity'] + self.channels] = intensities
        self.radius = radius
        self.overlap_threshold = overlap_threshold
        self.balance_threshold = balance_threshold

    def add_codebook(self,
                     codebook_path: str,
                     channel_info: str,
                     number_of_channels: int) -> None:
        """Load codebook for decoding
        Args:
            codebook_path (str): Path to codebook file
            channel_info (str): Path to channel info file
            number_of_channels (int): Number of spot channels
        """
        codebook = Codebook(codebook_path,channel_info,number_of_channels)
        self.codebook = codebook.codebook

    def decode(self) -> None:
        """Decode spots into barcodes for each nucleus"""
        final_barcodes = []
        nuclei = self.spots['nucleus'].unique()
        for nucleus in nuclei:
            spots1 = self.spots[self.spots['nucleus']==nucleus].copy().reset_index(drop=True)
            G = self.build_graph(spots1)
            components = nx.weakly_connected_components(G)
            barcodes = []
            for component in components:
                if len(component)>1:
                    barcodes.extend(self.decode_multi(spots1,G,component,nucleus))
                else:
                    if spots1.loc[list(component)]['true_single'].item():
                        barcodes.append(self.decode_single(spots1,list(component),nucleus))
            if len(barcodes)==0:
                continue
            columns = ['nucleus','y','x'] + self.channels + ['barcode']
            df = pd.DataFrame(np.array(barcodes),columns=columns)
            new_df = []
            for _,group in df.groupby('barcode'):
                coords = group[['y','x']].values
                clustering = DBSCAN(eps=self.radius,min_samples=1).fit(coords)
                group['cluster'] = clustering.labels_
                for _,cluster_group in group.groupby('cluster'):
                    if len(cluster_group)>1:
                        new_df.append(cluster_group.mean().values)
                    else:
                        new_df.append(cluster_group.iloc[0].values)
            columns = ['nucleus','y','x'] + self.channels + ['barcode','cluster']
            new_df = pd.DataFrame(np.stack(new_df),columns=columns).drop(columns='cluster')
            final_barcodes.append(new_df)
        final_barcodes = pd.concat(final_barcodes).reset_index(drop=True)
        dict1 = {'nucleus':'uint16','y':'float32','x':'float32','barcode':'uint32'}
        dict2 = {channel:'float32' for channel in self.channels}
        final_barcodes = final_barcodes.astype(dict1|dict2)
        final_barcodes['gene'] = final_barcodes['barcode'].apply(self.get_gene_barcode)
        path_to_save = Path.cwd()/f'{self.image_name}_spots_decoded.csv'
        final_barcodes.to_csv(path_to_save,index=False)
        print(f'Decoded spots are saved at {path_to_save}.')

    def get_gene_barcode(self,barcode: int) -> str:
        """Returns gene name for given channel combination
        Args:
            barcode (int): Channel combiantion
        Returns:
            str: gene name
        """
        if barcode in self.codebook:
            return self.codebook[barcode]
        else:
            return 'not in codebook'


    def build_graph(self,df: pd.DataFrame) -> nx.Graph:
        """Build graph of connected spots
        Args:
            df (pd.DataFrame): Table of spots
        Returns:
            nx.Graph: Graph connecting spots
        """
        coords = df[['y','x']].values
        tree = cKDTree(coords)
        neighbors = tree.query_ball_point(coords,r=self.radius)
        G = nx.DiGraph()
        G.add_nodes_from(list(df.index))
        for i,n in enumerate(neighbors):
            if len(n)>1:
                main_vector = df.loc[i][self.channels].values.astype('float')
                main_idx = int(re.search(r'\d+',df.loc[i]['channel']).group())-1
                main_channel = df.loc[i]['channel']
                main_coords = df.loc[i][['y','x']].values.astype('float')
                for m in n:
                    if m!=i:
                        edge = (i,m)
                        second_vector = df.loc[m][self.channels].values.astype('float')
                        second_idx = int(re.search(r'\d+',df.loc[m]['channel']).group())-1
                        second_channel = df.loc[m]['channel']
                        second_coords = df.loc[m][['y','x']].values.astype('float')
                        if second_channel==main_channel:
                            continue
                        # 80% overlap
                        overlap = np.divide(main_vector,second_vector,out=np.zeros_like(second_vector),where=second_vector!=0)
                        if overlap[second_idx]>self.overlap_threshold:
                            # intensity balance
                            balance = main_vector/main_vector.max() if main_vector.max()!=0 else main_vector
                            if np.all(balance[np.array([main_idx,second_idx])]>self.balance_threshold):
                                G.add_edge(edge[0],edge[1],distance=pdist(np.stack([main_coords,second_coords]),'euclidean'))
        return G
    
    def decode_single(self,
                      df: pd.DataFrame,
                      idx: int,
                      nucleus: int) -> list:
        """Decode single spots
        Args:
            df (pd.DataFrame): Table of spots
            idx (int): Spot index
            nucleus (int): Nucleus ID
        Returns:
            list: List of nucleus ID, coordinates, channel intensities, barcode
        """
        bc = df.loc[idx]['channel'].item()
        bc = int(re.search(r'\d+',bc).group())
        columns = ['y','x'] + self.channels
        return [nucleus]+list(df.loc[idx][columns].values.mean(axis=0))+[bc]
    
    def decode_multi(self,
                     df: pd.DataFrame,
                     G: nx.Graph,
                     indices: set,
                     nucleus: int) ->list:
        """Decode group of connected spots
        Args:
            df (pd.DataFrame): Table of spots
            G (nx.Graph): Graph of connected spots
            indices (set): Spot indices
            nucleus (int): Nucleus ID
        Returns:
            list: List of nucleus ID, coordinates, channel intensities, barcode
        """
        barcodes = []
        H = G.subgraph(indices).copy()
        edges = list(H.edges(data=True))
        edges = sorted(edges,key=lambda x: x[2]['distance'])
        spot_dict = {}
        for node in list(H.nodes):
            partners = list(H.successors(node))+[node]
            if len(partners)==1:
                if df.loc[partners]['true_single'].item():
                    spot_dict[node] = self.decode_single(df,partners,nucleus)
            else:
                bc = sorted(df.loc[partners]['channel'].tolist())
                bc = [re.search(r'\d+',bcch).group() for bcch in bc]
                bc = np.unique(bc)
                bc = int("".join(map(str,bc)))
                # optional condition variant to only include valid barcodes from codebook
                # if bc in self.codebook: # if I leave this, I have to think about the possibility that maybe no nodes remains in the graph
                #     columns = ['y','x'] + self.channels
                #     spot_dict[node]=[nucleus]+list(df.loc[partners][columns].values.mean(axis=0))+[bc]
                # else:
                #     H.remove_node(node)
                columns = ['y','x'] + self.channels
                spot_dict[node]=[nucleus]+list(df.loc[partners][columns].values.mean(axis=0))+[bc]
        for edge in edges:
            if not H.has_edge(edge[0],edge[1]):
                continue
            m1 = df.loc[edge[0]][self.channels].values.astype('float').max()
            m2 = df.loc[edge[1]][self.channels].values.astype('float').max()
            if m1>m2:
                H.remove_node(edge[1])
            elif m2>m1:
                H.remove_node(edge[0])
            else:
                H.remove_node(edge[1])
        for node in list(H.nodes):
            if node in spot_dict:
                barcodes.append(spot_dict[node])
        return barcodes


if __name__ == '__main__':
    # def parse arguments
    def validate_path(file_path):
        if not Path(file_path).exists():
            raise argparse.ArgumentTypeError(f"File not found: {file_path}")
        return Path(file_path)

    def parse_arguments():
        parser = argparse.ArgumentParser(description="Decoding single nuclei")
        parser.add_argument("-im","--image_name",type=str,required=True,
                            help='Image name.')
        parser.add_argument("-s","--spots", type=validate_path,required=True,
                            help='Path to spots')
        parser.add_argument("--codebook", type=validate_path,required=True,
                            help='Path to codebook')
        parser.add_argument("--channel_info", type=validate_path,required=True,
                            help='Path to channel info')
        parser.add_argument("--noc", type=int,required=False,default=6,
                            help='Number of spot channels.')
        parser.add_argument("--search_radius", type=int,required=False,default=1,
                            help='Serach radius for decoding.')
        parser.add_argument("--overlap_threshold", type=float,required=False,default=0.7,
                            help='Overlap of channels required to be multi color barcodes.')
        parser.add_argument("--balance_threshold", type=float,required=False,default=0.3,
                            help='Balance between channels intensities required to be multi color barcodes.')
        args = parser.parse_args()

        if args.noc<2:
            parser.error("Number of channels must be at least 2.")
        if args.search_radius<1:
            parser.error("Search radius must be at least 1.")
        if args.overlap_threshold<=0 and args.overlap_threshold>=1:
            parser.error("Overlap must be between 0 and 1.")
        if args.balance_threshold<=0 and args.balance_threshold>=1:
            parser.error("Balance must be between 0 and 1.")
        return args

    args = parse_arguments()

    decoding = Decoding(args.image_name,args.spots,args.noc,
                        args.search_radius,args.overlap_threshold,
                        args.balance_threshold)
    decoding.add_codebook(args.codebook,args.channel_info,args.noc)
    decoding.decode()

