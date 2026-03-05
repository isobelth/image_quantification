process CALCULATETHRESHOLDS {
    
    label 'medium'
    container 'registry.git.embl.org/felix.schneider1/podmanimages/base_image:v1.3.0'
    publishDir "${params.outdir}/results", pattern: "normalization_threshold.csv", mode: 'copy' , overwrite: true


    input:
    path stats_csvs

    output:
    path "normalization_threshold.csv", emit: threshold


    script:
    def pythonScript = "/project/src/python/CalculateThresholds.py"
    def args = task.ext.args ?: ''
    """
    python ${pythonScript} --nuclei_stats ${stats_csvs} $args
    """
}

