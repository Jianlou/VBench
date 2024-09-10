#!/bin/bash
HOME_DIR=${0%/*}
cd ${HOME_DIR}
HOME_DIR=$(pwd)
EXP=${1:-vbench}
VIDEO_PATH=${2:-/mnt/afs_1/sijianlou/code/VGen/data/instructvideo/generated/instructvideo_infer_UNetSD_t2v_webvid_LoRA_webvid_ddim20_in-domain_lora26k_vbench_946}
# DIMENSION=${3:-subject_consistency background_consistency temporal_flickering motion_smoothness dynamic_degree aesthetic_quality imaging_quality object_class multiple_objects human_action color spatial_relationship scene temporal_style appearance_style overall_consistency}
DIMENSION=${3:-imaging_quality overall_consistency}
# DIMENSION=${3:-object_class}
DATETIME=$(date '+%Y-%m-%d-%H:%M:%S')
echo ${HOME_DIR}
echo ${EXP}
echo ${DATETIME}
echo ${VIDEO_PATH}
echo ${DIMENSION}
srun --partition-id share-a \
    --workspace-id d08f360b-7f9c-4eb1-bfb3-155fdad18726 \
    --framework pt \
    --job-name ${EXP} \
    --resource N3lS.Ii.I60.1 \
    --distributed StandAlone \
    --output run_${DATETIME}.log \
    --nodes 1 \
    --priority NORMAL \
    --container-image registry.cn-sh-01.sensecore.cn/devsft-ccr/ubuntu20.04_cuda11.8_vgen:v3.0.1 \
    --container-mounts 4ba8dc8e-52e5-11ee-82fd-de3a99f44f33:/mnt/afs_1 \
    bash -c "cd "${HOME_DIR}"; source /usr/local/lib/miniconda3/bin/activate vbench;TRANSFORMERS_CACHE="/mnt/afs_1/sijianlou/code/VBench/vbench_models/huggingface/transformers" VBENCH_CACHE_DIR="/mnt/afs_1/sijianlou/code/VBench/vbench_models" python evaluate.py --videos_path ${VIDEO_PATH} --dimension ${DIMENSION} --load_ckpt_from_local True; sleep 1d"
