// Top and bottom image directories to be paired
#@ File (label = "Top directory", style = "directory") top_directory
#@ File (label = "Bottom directory", style = "directory") bottom_directory
#@ File (label = "Output directory", style = "directory") output
#@ String (label = "File initial_suffix", value = ".tif") initial_suffix

processFolder(top_directory);

function processFolder(top_directory) {
    list1 = getFileList(top_directory);
    list1 = Array.sort(list1);

    list2 = getFileList(bottom_directory);
    list2 = Array.sort(list2);

    print("Top folder files:", list1.length, " | Bottom folder files:", list2.length);

    // Exit if the "top" and "bottom" folders are of unequal length
    if (list1.length != list2.length) {
        print("Error: Number of images in folders do not match!");
        exit;
    }
    for (i = 0; i < list1.length; i++) {
        path1 = top_directory + File.separator + list1[i];
        path2 = bottom_directory + File.separator + list2[i];
        // Recursively process subdirectories (if any)
        if (File.isDirectory(path1)) {
            processFolder(path1);
        }
        if (File.isDirectory(path2)) {
            processFolder(path2);
        }
        // If both files match the suffix, process them
        if (endsWith(list1[i], initial_suffix) && endsWith(list2[i], initial_suffix)) {
            processFile(top_directory, bottom_directory, output, list1[i], list2[i]);
        }
    }
}

function processFile(top_directory, bottom_directory, output, topfile, bottomfile) {
    open(top_directory + File.separator + topfile);
    toptitle = getTitle();
    open(bottom_directory + File.separator + bottomfile);
    bottomtitle = getTitle();

    print("Combining:", toptitle, " + ", bottomtitle);

    // Concatenate the two stacks (assumes same size, frame count)
    run("Concatenate...", "image1=" + toptitle + " image2=" + bottomtitle);

    // Set display range and apply LUT (optional)
    setMinAndMax(0, 1);
    run("Apply LUT", "stack");

    // Rename output file by replacing "top" with "combined"
    newtitle = replace(toptitle, "top", "combined");

    saveAs(".tiff", output + File.separator + newtitle);
    close("*");
}
