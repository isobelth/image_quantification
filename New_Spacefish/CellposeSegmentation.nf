process CELLPOSE {

    errorStrategy 'retry'
    maxRetries 1

    tag "${image_name}"
    
    label 'segmentation'
    container "registry.git.embl.org/felix.schneider1/podmanimages/cellpose4:v4.0.6"

    input:
    tuple val(image_name),path(image_path),path(model)

    output:
    tuple val(image_name),path("*_masks.tif"), emit: initial_mask

    script:
    def pythonScript = "/project/src/python/CellposeSegmentation.py"
    def args = task.ext.args ?: ''
    """
    python3 ${pythonScript} -im "${image_name}" -f ${image_path} -m ${model} $args
    """
}
