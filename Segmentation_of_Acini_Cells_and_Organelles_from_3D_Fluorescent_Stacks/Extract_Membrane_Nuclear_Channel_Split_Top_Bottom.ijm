#@ File (label = "Input directory", style = "directory") input
#@ File (label = "Output directory", style = "directory") output
#@ String (label = "File suffix to match", value = ".tif") initial_suffix
#@ int (label = "Number of channels", value = 4) num_channels
#@ int (label = "First channel to keep", value = 1) ch1
#@ int (label = "Second channel to keep", value = 4) ch2

processFolder(input);

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

function processFile(input, output, file) {
    fullPath = input + File.separator + file;
    print("🔄 Processing:", fullPath);
    open(fullPath);
    origTitle = getTitle();

    total_slices = nSlices();
    slices_per_channel = total_slices / num_channels;

    if (slices_per_channel != floor(slices_per_channel)) {
        print("⚠️ Skipping due to inconsistent slice count:", total_slices, "not divisible by", num_channels);
        close();
        return;
    }

    // Step 1: Split Channels
    run("Split Channels");
    wait(500);

    title_ch1 = "C" + ch1 + "-" + origTitle;
    title_ch2 = "C" + ch2 + "-" + origTitle;

    if (!isOpen(title_ch1) || !isOpen(title_ch2)) {
        print("❌ Channels not found:", title_ch1, "or", title_ch2);
        close("*");
        return;
    }

    // Step 2: Merge Selected Channels
    run("Merge Channels...", "c1=[" + title_ch1 + "] c2=[" + title_ch2 + "] create");
    wait(300);
    rename("merged");

    if (!isOpen("merged")) {
        print("❌ Merge failed. Skipping.");
        close("*");
        return;
    }

    // Step 3: Duplicate Top Half
    selectWindow("merged");
    h = round(slices_per_channel / 2);
    run("Duplicate...", "duplicate slices=1-" + h);
    rename("top_" + file);
    print("💾 Saving top half:", "top_" + file);
    saveAs("Tiff", output + File.separator + "top_" + file);

    // Step 4: Duplicate Bottom Half
    hp1 = h + 1;
    selectWindow("merged");
    run("Duplicate...", "duplicate slices=" + hp1 + "-" + slices_per_channel);
    rename("bottom_" + file);
    print("💾 Saving bottom half:", "bottom_" + file);
    saveAs("Tiff", output + File.separator + "bottom_" + file);

    close("*");
}
