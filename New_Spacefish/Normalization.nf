process NORMALIZATION {
    
    tag "${meta.name}"

    label 'high'
    publishDir "${params.outdir}/results/${meta.name}", pattern: "*normalised.tif", mode: 'copy' , overwrite: true
    container 'registry.git.embl.org/felix.schneider1/podmanimages/base_image:v1.3.0'

    input:
    tuple val(meta), path(image_path)
    path threshold_csv
   
    output:
    tuple val(meta.name),path("*normalised.tif"), emit: normalized_nuclei
    tuple val(meta.name),path("nuclei_stats_normalized_*.csv"), emit: nuclei_stats_norm


    script:
    def pythonScript = "/project/src/python/Normalization.py"
    def args = task.ext.args ?: ''
    """
    python ${pythonScript} -im "${meta.name}" -f ${image_path} -t ${threshold_csv} $args
    """
}