process DECODING {
    tag "${image_name}"
    
    label 'low'
    publishDir "${params.outdir}/results/${image_name}", pattern: "*spots_decoded.csv", mode: 'copy' , overwrite: true
    publishDir "${params.outdir}/results/${image_name}", pattern: "*filtered_spots.csv", mode: 'copy' , overwrite: true
    container 'registry.git.embl.org/felix.schneider1/podmanimages/scikit_learn:1.7.0'

    input:
    tuple val(image_name),path(spot_csv)
    path codebook
    path channel_info

    output:
    tuple val(image_name),path("*spots_decoded.csv"), emit: decoded_spots
    tuple val(image_name),path(spot_csv), emit: normalized_spots


    script:
    def pythonScript = "/project/src/python/Decoding.py"
    def args = task.ext.args ?: ''
    """
    python ${pythonScript} -im "${image_name}" -s ${spot_csv} --codebook ${codebook} --channel_info ${channel_info} $args
    """
}
