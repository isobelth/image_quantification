process COUNTTABLE {
    tag "${image_name}"
    
    label 'low'
    publishDir "${params.outdir}/results/${image_name}", pattern: "*count_table.csv", mode: 'copy' , overwrite: true
    container 'registry.git.embl.org/felix.schneider1/podmanimages/base_image:v1.3.0'

    input:
    tuple val(image_name),path(decoded_spots),path(nucleus_stats)
    path codebook

    output:
    tuple val(image_name),path("*count_table.csv"), emit: count_table


    script:
    def pythonScript = "/project/src/python/CountTable.py"
    """
    python ${pythonScript} -im "${image_name}" -ds ${decoded_spots} -c ${codebook} --nucleus_stats ${nucleus_stats}
    """
}