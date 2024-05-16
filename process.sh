prompt="a blonde girl in black dress"
image_id="aria1"

# --------- Stage 0 preprocess --------- # 
image_path="images/$image_id.jpg"
python3 preprocess_image.py "$image_path" --recenter

# --------- Stage 1 (NeRF) --------- # 
image_path="images/${image_id}_rgba.png"
echo "--------- Stage 1 (NeRF) ---------"
python launch.py --config configs/dreamcraft3d-coarse-nerf.yaml --train system.prompt_processor.prompt="$prompt" data.image_path="$image_path"

# echo "--------- Stage 1 (NeUS) ---------"
# ckpt=outputs/dreamcraft3d-coarse-nerf/${prompt// /_}@LAST/ckpts/last.ckpt
# python launch.py --config configs/dreamcraft3d-coarse-neus.yaml --train system.prompt_processor.prompt="$prompt" data.image_path="$image_path" system.weights="$ckpt" data.height=128 data.width=128 data.random_camera.height=128 data.random_camera.width=128

# # --------- Stage 2 (Geometry Refinement) --------- # 
# echo "--------- Stage 2 (Geometry Refinement) ---------"
# ckpt=outputs/dreamcraft3d-coarse-neus/${prompt// /_}@LAST/ckpts/last.ckpt
# python launch.py --config configs/dreamcraft3d-geometry.yaml --train system.prompt_processor.prompt="$prompt" data.image_path="$image_path" system.geometry_convert_from="$ckpt" 

# # --------- Stage 3 (Texture Refinement) --------- # 
# echo "--------- Stage 3 (Texture Refinement) ---------"
# ckpt=outputs/dreamcraft3d-geometry/${prompt// /_}@LAST/ckpts/last.ckpt
# python launch.py --config configs/dreamcraft3d-texture.yaml --train system.prompt_processor.prompt="$prompt" data.image_path="$image_path" system.geometry_convert_from="$ckpt" 

# # --------- Stage 4 (Mesh Exporter) --------- # 
# ckpt=outputs/dreamcraft3d-texture/${prompt// /_}@LAST/ckpts/last.ckpt
# parsed_config="outputs/dreamcraft3d-texture/${prompt// /_}@LAST/configs/parsed.yaml"
# python3 launch.py --config "$parsed_config" --export --gpu 0 resume="$ckpt" system.exporter_type=mesh-exporter system.exporter.context_type=cuda