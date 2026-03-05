process CONCATNUCLEISTATS {

    tag "${image_name}"

    label 'low'
    container 'registry.git.embl.org/felix.schneider1/podmanimages/base_image:v1.3.0'
    publishDir "${params.outdir}/results/${image_name}", pattern: "*_NucleiStats*.csv", mode: 'copy', overwrite: true

    input:
    tuple val(image_name), path(nucleus_stats)
    val datatype

    output:
    tuple val(image_name), path("*_NucleiStats*.csv"), emit: concat_nuclei_stats

    script:
    def pythonScript = "/project/src/python/ConcatNucleiStats.py"
    """
    python ${pythonScript} -im "${image_name}" -f ${nucleus_stats} --datatype ${datatype}
    """
}