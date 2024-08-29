#!/bin/bash

# Define variables
IMAGE_PATTERN="/cns/jp-d/home/gchips-isp-algo-prod/datasets/video/scene79_park_walking/hdrnet_output_*.jpg"
OUTPUT_DIR="/cns/jp-d/home/gchips-isp-algo-prod/poyukao/gen_ai/diffusion/test_bld_video_xm"

# Run the command
gxm hardware/gchips/isp/ati/gen_ai/diffusion/xmanager/launcher.py \
  -- \
  --task="image_editing" \
  --config=hardware/gchips/isp/ati/gen_ai/diffusion/configs/image_editing.py \
  --config.image_pattern="${IMAGE_PATTERN}" \
  --config.output_dir="${OUTPUT_DIR}" \
  --cell="mg" \
  --job="PROD" \
  --xm_resource_pool="cml" \
  --xm_resource_alloc="cml/cml-shared-ml-user" \
  --noxm_monitor_on_launch
