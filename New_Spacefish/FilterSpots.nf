process FILTERSPOTS {

    label 'medium'
    container 'registry.git.embl.org/felix.schneider1/podmanimages/scikit_learn:1.7.0'
    publishDir "${params.outdir}/results", pattern: "spot_normalization_threshold.csv", mode: 'copy' , overwrite: true

    input:
    path spots_csv

    output:
    path "Filtered_Spots.csv", emit: filtered_spots
    path "spot_normalization_threshold.csv", emit: spot_normalization_threshold

    script:
    def pythonScript = "/project/src/python/FilterSpots.py"
    def args = task.ext.args ?: ''
    """
    python ${pythonScript}  -s ${spots_csv} $args
    """
}