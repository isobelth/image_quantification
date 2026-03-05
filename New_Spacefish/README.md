# Project
SPACE_FISH is a method which can be used to multiplex up to 18 genes in on experiment.
The principal is to use 6 color channels to encode the 18 genes. So each gene is either encoded with only one or with two color channels.
This pipeline aims at processing the data, so that the output will be the decoded spots for each nucleus and a gene count table for each image.

# How to run the pipeline
## Setup on HPC
1. Go to you scratch directory
`cd /scratch/$USER`
2. Clone repo
`git clone -b main https://git.embl.de/felix.schneider1/space-fish.git`

## Run the pipeline
1. Create an analysis folder in your group directory or on /scratch
2. Create a .csv file in this directory listing the files and scenes to process (see input file format)
3. Copy params.config file into this directory and change the parameters to your needs (leave unused parameters to default)
4. Copy RunPipeline.sh into this directory and modify the relevant lines
5. submit RunPipeline.sh to the SLURM scheduler
`sbatch path/to/file/RunPipeline.sh`

# Requisite for using this pipeline
## Image data
- 3D
- 7-8 channels in following order:
    - 6 color channels with spots
    - 1 DAPI image for nuclei
    - (optional) 1 DIC image for cell shapes
## Codebook
- channel_info (.txt):
    - one line per channel
    - first column channel number
    - second column fluorophore
    - without header
- codebook (.csv):
    - header: gene,color1,color2,color3,color4
    - one line per gene
    - color entries should match fluorophore names in channel_info
    - gene encoding: one or two fluorophore per gene
    - blank/false positive encoding: three or four fluorophore per blank/false positive
## Input files
- input.csv:
    - four columns:
        - image_name: Either 'auto' for auto detection or actual name
        - scene_id: Index of image scene to process (zero-based)
        - normalisation: Should the image be included to calculate the value for channel normalisation (true,yes,1)
        - save_brightfield: Should a brightfield image be saved (true,yes,1)
        - path: Absolute path to the image file
- censoring_input.csv:
    - three columns:
        - image_name: actual image name (has to be the same as in input.csv)
        - path: Absolute path to the censoring regions file
- censor_area.csv: (Example file format)
    - columns: z1,z2,x_z1_1,x_z1_2,y_z1_1,y_z1_2,x_z2_1,x_z2_2,y_z2_1,y_z2_2
    - if censoring a 2D image keep the same columns,enter any number for z1/z2 and repeat the boundaries for x and y


**Important**: The image_name have to be the exact same in input.csv and censoring_input.csv, otherwise it will not work. Please do not use space or weird symbol in the image name. Best practise is to use lower/upper case letters, numbers and underscores

# Pipeline output
- tabular output
    - \*_filtered_spots.csv
        - filtered detected raw spots in nucleus coordinate system
    - \*_nucleiStats_normalized.csv
        - Statistics on nucleus intensity after normalization
    - \*_NucleiStats.csv
        - Statistics on nucleus intensity and its location in global coordinate system
    - \*_spots_concat.csv
        - All (raw) detected spots in nucleus coordinate system
    - \*_spots_decoded.csv
        - Decoded barcodes (spots) in nucleus coordinate system
    - \*_count_table.csv
        - Count matrix of barcodes per nucleus
    - normalization_threshold.csv
        - Intensity normalisation values
- image output
    - \*-nucleus\*.tif
        - 2D cropped image of nucleus including all color channels, DAPI and nucleus segmentation mask
    - \*-nucleus\*-normalised.tif
        - 2D cropped normalised image of nucleus including all color channels, DAPI and nucleus segmentation mask
    - \*-seg_mask.tif
        - Nuclei segmentation mask of original image

# Pipeline steps:
1. SPLITIMAGE (SplitImage.py): Splits image into three different channels: spot channels, DAPI channel, brightfield image (optional if available), do maximum z projection (if chosen)
    - input: image name, image path
    - args: number_of_spotchannels, dapi_channel,brightfield_channel,save_brightfield,scale_factor_xy,scale_factor_z,cellpose_anisotropy,do_z_projection
    - output: brightfield image (fi available), image_name+dapi_channel,image_name+spotchannel
    - publish: brightfield image (if available)
2. PREPROCESSDAPI (PreProcessDAPI.py): Downsample image and gaussian blurs DAPI image
    - input: image name image path
    - args: scale_factor_xy,scale_factor_z,dapi_sigma_xy,dapi_sigma_z
    - output: image_name+processed_dapi
3. CELLPOSE (CellposeSegmentation.py): Segmentation of nuclei with cellpose
    - input: image name, image path
    - args: cellpose_seg_model,cellpose_nucleus_diameter,do_z_projection,cellpose_anisotropy,cellpose_dP_smooth,cellpose_min_size
    - output: image name, segmentation mask
4. SEGMENTATIONCLEANUP (SegmentationCleanUp.py): Upscales segmentation mask to original size and removes border nuclei (optional)
    - input: image name, image path
    - args: scale_factor_xy,scale_factor_z,exclude_border_nuclei
    - output: image name, segmentation mask
    - publish: segmentation mask
5. CROPNUCLEI (CropNuclei.py): Crops image into its individual nuclei for further processing
    - input: image name, dapi image, spotchannel image, segmentation mask, censoring regions
    - args: crop_buffer,censor,convexity_threshold,do_z_projection
    - output: image_name+nuclei_locations.csv, image_name+crop_nucleus.tif
    - publish: nuclei coordinates
6. REFINEMASK (RefineMask.py): Refine segmentation mask after cropping (optional)
    - input: image_name, images
    - args: refine_seg_model,refine_nucleus_diameter,convexity_threshold
    - output: image name, refined crops
7. NUCLEUSSTATS (NucleusStats.py): 
    - input: image name, images
    - output: image name, nucleus statistics
    - publish: cropped nuclei
8. CONCATRAW (ConcatNucleiStats.py): Concatenate statistics on nuclei
    - input: image_name+nuclei_statistics, raw or normalised
    - output: image name, concatenated statistics
    - publish: concatenated statistics
9. CALCULATETHRESHOLDS (CalculateThresholds.py): Calculate intensity thresholds for normalisation
    - input: nuclei statistics
    - args: default_p99
    - output: normalization threshold
    - publish: normalization threshold
10. NORMALIZATION (Normalization.py): Normalise signal intensity across all nuclei and images and re-calculates the nucleus statistics
    - input: image_name+image_path, thresholds
    - args: background_sigma,smoothing_sigma
    - output: image_name+normalised_nuclei,image_name+normalised_nuclei_stats
    - publish: normalised cropped nuclei
11. CONCATNORM (ConcatNucleiStats.py): Concatenate statistics on nuclei
    - input: image_name+nuclei_statistics, raw or normalised
    - output: image name, concatenated statistics
    - publish: concatenated statistics
12. SPOTIFLOW (SpotDetection.py): Detect spots inside the nucleus
    - input: image name, image path
    - args: spotiflow_model,spotiflow_min_spot_distance
    - output: image_name, detected spots
13. CONCATSPOTS (ConcatSpots.py): Concat spots on sample level
    - input: image name, spots
    - output: image name, concatenated spots
    - publish: concatenated spots
14. FILTERSPOTS (FilterSpots.py): Filter spots either globally or individually
    - input: spots
    - args: do_global_normalization,do_global_filtering
    - output: table of filtered spots
    - publish: threshold and kneepoints for spots
15. DECODING (Decoding.py): Decode spots into barcodes
    - input: image_name+spots, codebook, channel_info
    - args: decoding_search_radius,number_of_spotchannels,decoding_overlap_threshold,decoding_balance_threshold
    - output: image_name+decoded_spots, image_name+spots
    - publish: decoded spots, filtered and normalised spots
16. COUNTTABLE (CountTable.py): Convert barcodes into count matrix
    - input: image_name+decoded_barcode+nucleus_stats, codebook
    - output: gene count matrix
    - publish: gene count matrix



# Parameter file (params.config)
- *input*: Absolute path to the input csv table
- *censor_input*: Absolute path to table of paths linking to files that contain the censoring regions, names have to match the image_name
- *codebook*: Absolute path to the codebook
- *channel_info*: Absolute path to the channel infos
- *outdir*: Absolute path to output directory where to save the pipeline output
- *number_of_spotchannels*: Number of spot channels, default to 6
- *dapi_channel_index*: Zero-based index of DAPI channel, default to 6
- *brightfield_channel_index*: Zero-based index of DAPI channel, default to 7
- *do_z_projection*: Should image be maximum z projected before processing. Defaults to false.
- *scale_factor_xy*: Scale factor to down- and upsample DAPI for processing in xy, default is 3
- *scale_factor_z*: Scale factor to down- and upsample DAPI for processing in z, default is 3. Will be ignored when do_z_projection=true
- *dapi_sigma_xy*: Smoothing factor for DAPI for processing in xy, default is 3
- *dapi_sigma_z*: Smoothing factor for DAPI for processing in z, default is 3. Will be ignored when do_z_projection=true
- *cellpose_seg_model*: Cellpose segmentation model to use for segmentation
- *cellpose_min_size*: Minimum size of segmented objects, default is 500
- *cellpose_anisotropy*: Anisotropy in z of the image. Default is 4. Will be ignored when do_z_projection=true
- *cellpose_stitch_threshold*: Stitching threshold for stitching 2D masks into 3D masks. Default is 0.1
- *censor*: Should nuclei excluded based on provided censor regions. Default is false. Only turn to true if do_z_projection=false
- *exclude_border_nuclei*: Should border nuclei be excluded for further downstream processing, Default is true
- *refine_seg_model*: Cellpose segmentation model to use for segmentation
- *default_p99*: A default value for the 99% percentile of nucleus intensities for cases where single stains are happening.
- *do_global_nucleus_normalization*: Should nucleus intensity normalisation be done over all images or on each individually
- *background_sigma*: Sigma value to estimate background for background subtraction. Default is 9
- *smoothing_sigma*: Sigma value to smooth image after background subtraction and before further processing. Default is 2
- *crop_buffer*: How much buffer in z should be added to the spot channels before doing maximum z projection. Default is 5
- *convexity_threshold*: Threshold to filter for convexity. Every nucleus with a convexity lower then the threshold will be discarded. Default it 0.7
- *additional_cleanup*: Option to do some more mask removal before cropping based on mask size, intensity and shape
- *spotiflow_model*: Spotiflow model for spot detection. As images are 2D only 2D models will work
- *spotiflow_min_spot_distance*: Minimu distance between two spots so that they will be considered as two spots. Default is 3
- *do_global_normalization*: Should spot intensity normalisation be done over all images or on each individually
- *do_global_filtering*: Should spot filtering be done over all images or on each individually
- *decoding_search_radius*: Search radius in which spots are close enough to be considered part of one barcode
- *decoding_overlap_threshold*: Threshold for overlapping of two spots. Default is 0.7 If it is below spots are not considered overlapping
- *decoding_balance_threshold*: Threshold for spot intensity balancing. Default is 0.3 If value is below spots are not balance

# RunPipeline.sh
Please change the relevant lines like:
- SLURM job title
- your group affiliation
- your email address
- path to store the slurm output
- the folder where you downloaded this repo
- the path to your params.config file
- which cluster setting you need

```bash
#! /bin/bash
#SBATCH --job-name=SPACE-FISH_Pipeline
#SBATCH -A your_GROUP
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -t 0-04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@mail.de
#SBATCH -o path/to/slurm/output/slurm-%j.out

# Unload all modules
module purge

# Load Nextflow 24.10.0
module load Nextflow/24.10.0

# change to space-fish directory
cd /scratch/$USER/space-fish      # if your folder on scratch is not named as your user please change $USER to your actual folder name

# make dir for apptainer cache
mkdir -p .apptainer_cache

# Run pipeline
srun nextflow run main.nf -c path/to/params/config/file -profile apptainer,cluster   # comment when high memory is needed

# Run pipeline with high memory
# srun nextflow run main.nf -c path/to/params/config/file -profile apptainer,clusterHighMem    # uncomment for high memory run

```
