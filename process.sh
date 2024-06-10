image_id="youzi"
# prompt="a woman in a black coat holding a bat"

# --------- Stage 0 preprocess --------- # 
echo "--------- Stage 0 (Preprocess) ---------"
image_path_old="images/0530/${image_id}.png"
image_path="images/0530/${image_id}_rgba.png"
normal_path="images/0530/${image_id}_normal.png"
depth_path="images/0530/${image_id}_depth.png"
caption_path="images/0530/${image_id}_caption.txt"
rm ${image_path}
rm ${normal_path}
rm ${depth_path}
rm ${caption_path}
# --------- Background Removal --------- # 
python3 -m carvekit  -i ${image_path_old} -o ${image_path} \
    --device cuda --net basnet --seg_mask_size 320
python3 preprocess_image.py "$image_path" --recenter --do_caption
prompt=$(cat $caption_path)

# --------- Stage 1 (NeRF & NeUS) --------- # 
echo "--------- Stage 1 (NeRF) ---------"
python launch.py --config configs/dreamcraft3d-coarse-nerf.yaml --train system.prompt_processor.prompt="$prompt" data.image_path="$image_path"

echo "--------- Stage 1 (NeUS) ---------"
ckpt=outputs/dreamcraft3d-coarse-nerf/${prompt// /_}@LAST/ckpts/last.ckpt
python launch.py --config configs/dreamcraft3d-coarse-neus.yaml --train system.prompt_processor.prompt="$prompt" data.image_path="$image_path" system.weights="$ckpt" data.height=128 data.width=128 data.random_camera.height=128 data.random_camera.width=128

# --------- Stage 2 (Geometry Refinement) --------- # 
echo "--------- Stage 2 (Geometry Refinement) ---------"
ckpt=outputs/dreamcraft3d-coarse-neus/${prompt// /_}@LAST/ckpts/last.ckpt
python launch.py --config configs/dreamcraft3d-geometry.yaml --train system.prompt_processor.prompt="$prompt" data.image_path="$image_path" system.geometry_convert_from="$ckpt" 

# --------- Stage 3 (Texture Refinement) --------- # 
echo "--------- Stage 3 (Texture Refinement) ---------"
ckpt=outputs/dreamcraft3d-geometry/${prompt// /_}@LAST/ckpts/last.ckpt
python launch.py --config configs/dreamcraft3d-texture.yaml --train system.prompt_processor.prompt="$prompt" data.image_path="$image_path" system.geometry_convert_from="$ckpt" 

# --------- Stage 4 (Mesh Exporter) --------- # 
echo "--------- Stage 4 (Mesh Exporter) ---------"
LAST=$(find 'outputs/dreamcraft3d-texture/' -type d -name ${prompt// /_}* | sort -n | tail -n1)
ckpt=${LAST}/ckpts/last.ckpt
parsed_config=${LAST}/configs/parsed.yaml
python launch.py --config "$parsed_config" --export --gpu 0 resume="$ckpt" system.exporter_type=mesh-exporter system.exporter.context_type=cuda