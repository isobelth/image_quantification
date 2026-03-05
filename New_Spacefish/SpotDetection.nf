process SPOTIFLOW {

    label 'medium'
    container 'registry.git.embl.org/felix.schneider1/podmanimages/spotiflow:v1.3.2'
    
    input:
    tuple val(image_name),path(image_path)

    output:
    tuple val(image_name),path("*.csv"), emit: raw_spots, optional: true

    script:
    def pythonScript = "/project/src/python/SpotDetection.py"
    def args = task.ext.args ?: ''
    """
    python ${pythonScript} -im "${image_name}" -f ${image_path} $args
    """
}