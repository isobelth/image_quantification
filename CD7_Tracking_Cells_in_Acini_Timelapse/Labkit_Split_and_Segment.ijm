#@ File (label = "Input directory", style = "directory") input
#@ File (label = "Output directory", style = "directory") output
#@ String (choices={"1", "2", "3", "4"}, style="radioButtonHorizontal") nuclear_channel
#@ Boolean (label = "Do you have a GPU?") useGPU
#@ String (label = "File initial_suffix", value = ".tif") initial_suffix
#@ File (label = "Top Classifier", style = "file") top_classifier
#@ File (label = "Bottom Classifier", style = "file") bottom_classifier

nuclear_channel = parseInt(nuclear_channel);

processFolder(input);

// === FUNCTION: Process a folder and its subfolders ===
// Recursively searches for files ending with initial_suffix and processes them
function processFolder(input) {
    list = getFileList(input);
    list = Array.sort(list);
    for (i = 0; i < list.length; i++) {
        path = input + File.separator + list[i];
        if (File.isDirectory(path)) {
            processFolder(path);
        } else if (endsWith(list[i], initial_suffix)) {
            processFile(input, output, list[i]);
        }
    }
}

// === FUNCTION: Process a single file ===
function processFile(input, output, file) {
    // --- Open and split top/bottom of nuclear channel ---
    open(input + File.separator + file);
    getDimensions(width, height, channels, slices, frames);

    title = getTitle();
    totalSlices = round(slices); 
    midSlice = round(totalSlices / 2);
    midStr = "" + midSlice;
    midPlus1Str = "" + (midSlice + 1);
    totalStr = "" + totalSlices;

    print("Processing:", file, "| Mid =", midStr, "| Frames =", frames);

    // Duplicate top half of Z slices (slices 1 to midSlice)
    run("Duplicate...", "duplicate channels=" + nuclear_channel + " slices=1-" + midStr);
    top_title = getTitle();

    // Duplicate bottom half of Z slices (slices mid+1 to total)
    selectWindow(title);
    run("Duplicate...", "duplicate channels=" + nuclear_channel + " slices=" + midPlus1Str + "-" + totalStr);
    bottom_title = getTitle();

    // --- Loop over time points and segment each Z-half separately ---
    for (t = 1; t <= frames; t++) {
        // --- Process top Z stack at time t ---
        selectWindow(top_title);
        run("Duplicate...", "duplicate frames=" + t + " title=Top_T" + t);
        top_t_title = "Top_T" + t;
        selectWindow(top_t_title);
        run("Segment Image With Labkit", "segmenter_file=" + top_classifier + " use_gpu=" + useGPU);
        saveAs(".tiff", output + File.separator + "top_T" + t + "_" + file);
        close(top_t_title);

        // --- Process bottom Z stack at time t ---
        selectWindow(bottom_title);
        run("Duplicate...", "duplicate frames=" + t + " title=Bottom_T" + t);
        bottom_t_title = "Bottom_T" + t;
        selectWindow(bottom_t_title);
        run("Segment Image With Labkit", "segmenter_file=" + bottom_classifier + " use_gpu=" + useGPU);
        saveAs(".tiff", output + File.separator + "bottom_T" + t + "_" + file);
        close(bottom_t_title);
    }
    close("*");
}