# Label Generation Steps
PLEASE HELP WITH STEP 4 OF THIS PLAN!
1. Gather paired brightfield + fluorescent images
   - If you have more paired images, please let me know!
3. Use a deep learning model and sparse labelling to create a mask. Stack the images into a 3-channel stack:
   - Channel 0: Brightfield, Channel 1: Fluorescence, Channel 2: Mask
   - Images saved in `Z:\Bel\IMAGES_FOR_VASCUMAP_RETRAINING\fluorescent_cells_tifs\3_channel_images_to_curate`
4. Use the GUI in this repository to curate the masks 
5. (Hopefully!) use the brightfield and curated masks to train a deep learning model to segment vasculature directly from the brightfield images

<img src="README_images/label_curation_1.png" width="80%" />

# How to Use This Code

1. If you've ever used the permeability/placenta code, you already have VS code and the relevant environment installed, so you can skip ahead to step 7
2. Install Visual Studio (VS) Code from: https://code.visualstudio.com/
3. Install Miniconda using this installer: https://docs.conda.io/en/latest/miniconda.html
   - Make sure you check the "Add Miniconda to my PATH environment variable"
4. Open Visual Studio Code. On the top menu, select Terminal → New Terminal. When the terminal pops up, make sure it says Command Prompt and not PowerShell
   - You can check that Miniconda is installed by typing `conda --version` in the terminal. If you see a number, conda is working! Chat to Bel if you get an error here!
5. Click on the `environment.yml` file in this directory and press download. Save it to a designated Python folder. In VS code, go to File → Open Folder → Select your designated folder
6. In your terminal, write the following (pay attention to spaces!) and press Enter: `conda env create -f environment.yml`
   - Lots of stuff will appear in the terminal, allow it to install (you might have to type `a` and press enter to agree to the conditions
7. Download the label_curation.ipynb file from this directory and save it to your designated Python folder. Click on it in VS Code and select the newly installed terminal (nap-ij) in the top right
8. You might have to press `python environments` to see it listed
9. Press the `Run All` button at the top of the file. The GUI should open in a new window
    
# How to Use The GUI
1. Select a file here and press `Load image`. 
2. Updates/error/success messages print here. You can see when the image has been loaded or when any operations have been completed. An autosave operation happens every few minutes in the background so you don't lose your progress. 
   - A separate mask file is generated and only that one is saved. When you press "final save", the original mask is overwritten. If you have any issues (eg powercut) and you want to make sure your progress has been saved, speak to Bel! Hopefully I've thought of every possible catch to ensure no progress is lost.
3. The 3 image channels are shown on the bottom left. When you select the mask you can:
    - Change the opacity
    - Select and edit the mask using the eraser or paintbrush (these only work when the mask visibility (eye) is toggled on). If you change `n edit dim` to 3 you are altering all slices simultaneously (not advised!)
    - All vessels should have label = 1. You can change colour to aid visibility by pressing the button with two curved arrows
    - You can select whole regions to label as 1 (mask) or 0 (background) by using the polygon tool.
    - You can change the brush/eraser size by holding down Alt and moving your mouse right (larger) or left (smaller)
 4. With the brightfield/fluorescence layer selected (blue) you can change the layer contrast limits (like brightness and contrast in FIJI), colour, or gamma (brings out brights/darks depending on direction)
 5. You can toggle between 2D and 3D by pressing the square/cube button, or change the viewing axis by pressing the third and fourth buttons. Press Home to reset the view
 6. Scroll through the slices using this slider, or hold Ctrl+mouse wheel. To zoom in/out, use the mouse wheel, and hold down space and drag to move. Ctrl + Z undoes any action.

 I've added in a few functions to make the labelling more painless:
 7. `REPLACE SLICES`
    - This lets you label one slice well and then replace another slice with the well-curated layer
    - You can perform extra curation afterwards; just use your initial slice as a base!
 8. `INTERPOLATE`
    - If you curate a slice e.g. near the top and one in the middle, you can interpolate between them (and then curate more closely)
 9. `FADE OUT`
    - If you image black to black, this works like INTERPOLATE, but one slice is all black (no signal)
    - e.g. layer a slice near the top and one that's fully out of focus. Whatever z you enter into `to z` will be set to all black.
 10. `SAVE`
    - This overwrites the original label with your curated one (full stack). You can see when the save is successful by looking in box (2.)

<img src="README_images/gui_overview.png" width="80%" />

If you can think of any other functionality that would aid labelling, please let me know and I'm happy to code it up!

# Advised Workflow
1. Decide on an image to curate in `Z:\Bel\IMAGES_FOR_VASCUMAP_RETRAINING\fluorescent_cells_tifs\3_channel_images_to_curate`. If you need me to create base labels for a particular image, just let me know!
   - It would be good to get a range of people curating, so please don't just curate your own images
2. Add a suffix to the image name so it's clear someone is working on it e.g. `BEL_WIP`.
3. Open the GUI and select your chosen image. Curate it to your heart's content/until you lose the will to live
   - You don't have to finish every layer. You can just save, and someone else can continue (makes the end model robust!)
4. When you've pressed `Save final` (and seen the success message in the logger), check that your image has been saved (you can see the "last edited" time in the file explorer).
   - If it's not the current time, speak to Bel before you shut anything down to ensure no progress is lost (it shouldn't be!)
5. Remove your suffix from the image so someone else can work on it
6. If you finish a label, move the image into the `DONE` folder.
