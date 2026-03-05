process SPLITIMAGE {
    tag "${meta.name}"

    label 'medium'
    publishDir "${params.outdir}/results/${meta.name}", pattern: "*-brightfield.tif", mode: 'move', overwrite: true, enable: ({meta.save_brightfield}?.toString()?.toLowerCase() in ['1','yes','true'])
    container 'registry.git.embl.org/felix.schneider1/podmanimages/base_image:v1.3.0'
    
    input:
    tuple val(meta),path(image_path)

    output:
    path "*-brightfield.tif", emit: brightfield, optional: true
    tuple val(meta),path("*-dapi.ome.zarr"), emit: dapi
    tuple val(meta),path("*-spotchannel.ome.zarr") , emit: spots

    script:
    def pythonScript = "/project/src/python/SplitImage.py"
    def args = task.ext.args ?: ''
    def save_brightfield = meta.save_brightfield?.toString()?.toLowerCase() in ['1','yes','true']
    def brightfield_opt = save_brightfield ? "--save_brightfield" : ""
    """
    python ${pythonScript} -im ${meta.name} -f ${image_path} --scene ${meta.scene} $args $brightfield_opt
    """
}
