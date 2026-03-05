process CROPNUCLEI {
    tag "${meta.name}"

    label 'crop_nuclei'
    container 'registry.git.embl.org/felix.schneider1/podmanimages/base_image:v1.3.0'
    publishDir "${params.outdir}/results/${meta.name}", pattern: "*.csv", mode: 'copy' , overwrite: true
    publishDir "${params.outdir}/results/${meta.name}", pattern: "*_cropped_3D.ome.tif", mode: 'move', overwrite: true, enable: !params.do_z_projection
    publishDir "${params.outdir}/results/${meta.name}", pattern: "*-seg_mask.tif", mode: 'copy' , overwrite: true
    
    input:
    tuple val(meta),path(dapi),path(spotchannel),path(mask)

    output:
    tuple val(meta),path("*.csv"), emit: nuclei_locations
    tuple val(meta), path("*_cropped.tif"),    emit: crop_nuclei
    tuple val(meta), path("*-seg_mask.tif"),    emit: seg_mask
    path "*_cropped_3D.ome.tif", emit: crop_3D, optional: true
    
    script:
    def pythonScript = "/project/src/python/CropNuclei.py"
    def args = task.ext.args ?: ''
    """
    python ${pythonScript} -im "${meta.name}" -s ${spotchannel} -d ${dapi} -m ${mask} $args
    """
}