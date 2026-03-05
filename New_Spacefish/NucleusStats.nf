process NUCLEUSSTATS {
    tag "${meta.name}"

    label 'medium'
    publishDir "${params.outdir}/results/${meta.name}", pattern: "*.tif", mode: 'copy', overwrite: true
    container 'registry.git.embl.org/felix.schneider1/podmanimages/base_image:v1.3.0'
    
    input:
    tuple val(meta),path(images)

    output:
    tuple val(meta.name),path("*.csv"), emit: nucleus_stats
    tuple val(meta.name),path(images), emit: input_images
    
    script:
    def pythonScript = "/project/src/python/NucleusStats.py"
    def norm_flag = meta.norm?.toString()?.toLowerCase() in ['1','yes','true']
    def norm_opt = norm_flag ? "--flag" : ""
    """
    python ${pythonScript} -im "${meta.name}" -f ${images} $norm_opt
    """
}