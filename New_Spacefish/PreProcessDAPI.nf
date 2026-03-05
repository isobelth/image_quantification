process PREPROCESSDAPI {
    tag "${image_name}"

    label 'medium'
    container 'registry.git.embl.org/felix.schneider1/podmanimages/base_image:v1.3.0'
    
    input:
    tuple val(image_name),path(image_path)

    output:
    tuple val(image_name),path("*-dapi_processed.tif"), emit: dapi_processed

    script:
    def pythonScript = "/project/src/python/PreProcessDAPI.py"
    def args = task.ext.args ?: ''
    """
    python ${pythonScript} -im ${image_name} -f ${image_path} $args
    """
}
