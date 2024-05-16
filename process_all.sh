for img in images/*; do
    if [[ -f "$img" ]]; then  # 确保它是文件
        python3 preprocess_image.py "$img" --recenter
    fi
done