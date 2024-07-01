#!/bin/bash
# /home/mint/Dev/DiFaReli/difareli-faster/experiment_scripts/TPAMI/sh_to_envs/video_out_512

CUDA_VISIBLE_DEVICES=3 python demo_bg.py --input_fg /data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/valid/68364.jpg --input_bg /home/mint/Dev/DiFaReli/difareli-faster/experiment_scripts/TPAMI/sh_to_envs/video_out_512/src=68364.jpg_dst=65284.jpg/map_centered.mp4 --save_dir ./ic-light_out/src=68364.jpg_dst=65284.jpg/ --image_width 512 --image_height 512 --prompt "handsome man" && wait &&
CUDA_VISIBLE_DEVICES=3 python demo_bg.py --input_fg /data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/valid/69267.jpg --input_bg /home/mint/Dev/DiFaReli/difareli-faster/experiment_scripts/TPAMI/sh_to_envs/video_out_512/src=69267.jpg_dst=65262.jpg/map_centered.mp4 --save_dir ./ic-light_out/src=69267.jpg_dst=65262.jpg/ --image_width 512 --image_height 512 --prompt "beautiful woman"
