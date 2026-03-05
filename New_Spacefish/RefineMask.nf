process REFINEMASK {

    tag "${meta.name}"

    label 'segmentation'
    container 'registry.git.embl.org/felix.schneider1/podmanimages/cellpose4:v4.0.6'

    input:
    tuple val(meta), path(images),path(model)

    output:
    tuple val(meta), path("*.tif"), emit: crop_nuclei, optional: true

    script:
    def pythonScript = "/project/src/python/RefineMask.py"
    def args = task.ext.args ?: ''
    """
    python3 ${pythonScript} -im "${meta.name}" -f ${images} -m ${model} $args
    """
}

