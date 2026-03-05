process CONCATSPOTS {

    tag "${image_name}"

    label 'low'
    container 'registry.git.embl.org/felix.schneider1/podmanimages/base_image:v1.3.0'
    publishDir "${params.outdir}/results/${image_name}", pattern: "*_spots_concat.csv", mode: 'copy', overwrite: true

    input:
    tuple val(image_name), path(spots)

    output:
    tuple val(image_name), path("*_spots_concat.csv"), emit: concat_spots

    script:
    def pythonScript = "/project/src/python/ConcatSpots.py"
    """
    python ${pythonScript} -im "${image_name}" -f ${spots}
    """
}