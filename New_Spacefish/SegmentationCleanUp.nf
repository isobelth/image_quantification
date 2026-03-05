process SEGMENTATIONCLEANUP {
    tag "${image_name}"

    label 'medium'
    container 'registry.git.embl.org/felix.schneider1/podmanimages/base_image:v1.3.0'
    
    input:
    tuple val(image_name),path(image_path),path(censor_region)

    output:
    tuple val(image_name),path("*-seg_mask_cleaned.tif"), emit: seg_mask

    script:
    def pythonScript = "/project/src/python/SegmentationCleanUp.py"
    def args = task.ext.args ?: ''
    def do_censoring = censor_region.name != 'censor_dummy.txt'
    def censor_opt = do_censoring ? "--censor --censor_region ${censor_region}" : ""
    """
    python ${pythonScript} -im "${image_name}" -f ${image_path} $args $censor_opt
    """
}
