#!/bin/bash


# Folder containing the input images and masks
input_folder="inputs"

# Loop through files in the input folder
for file in "$input_folder"/*_image_condition.png; do

  # Extract the base filename (without extension)
  base_name="${file##*/}"
  base_name="${base_name%.*}"
  base_name="${base_name%.*}"

  # Construct the full paths for init_image, mask, and output_path
  init_image="$input_folder/${base_name}._image_condition.png"
  output_path="outputs/${base_name}._output.png"

  # Run the Python script with the extracted filenames
  python scripts/text_editing_IP2P.py \
      --prompt "starry night sky" \
      --init_image "$init_image" \
      --output_path "$output_path" \

done
