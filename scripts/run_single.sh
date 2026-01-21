#!/bin/bash

# Single experiment execution script
# Usage: ./scripts/run_single.sh <experiment_name>

EXP_KEY=$1

if [ -z "$EXP_KEY" ]; then
    echo "Usage: ./scripts/run_single.sh <experiment_name>"
    echo "Available experiments: exp_c"
    exit 1
fi

WORK_DIR="/home/songliyu/Documents/Depth-Anything-3"
IMAGE_DIR_560="$WORK_DIR/data/iphone11-frames-560"
CONFIG_DIR="$WORK_DIR/da3_streaming/configs/iphone11_exp"
OUTPUT_BASE="$WORK_DIR/output/iphone11-exp"

declare -A EXPERIMENTS=(
    ["exp_c"]="exp_c_chunk18_560.yaml:$IMAGE_DIR_560"
)

if [ -z "${EXPERIMENTS[$EXP_KEY]}" ]; then
    echo "Error: Unknown experiment name '$EXP_KEY'"
    exit 1
fi

IFS=':' read -r config_file img_dir <<< "${EXPERIMENTS[$EXP_KEY]}"

config_path="$CONFIG_DIR/$config_file"
output_dir="$OUTPUT_BASE/$EXP_KEY"

echo "========================================="
echo "Executing experiment: $EXP_KEY"
echo "========================================="
echo "Config: $config_path"
echo "Images: $img_dir"
echo "Output: $output_dir"
echo ""

cd "$WORK_DIR"

# Start GPU monitoring
bash scripts/monitor_gpu.sh "$EXP_KEY" &
MONITOR_PID=$!
echo "GPU monitoring PID: $MONITOR_PID"

# Record start time
START_TIME=$(date +%s)

# Execute prediction
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python da3_streaming/da3_streaming.py \
    --image_dir "$img_dir" \
    --config "$config_path" \
    --output_dir "$output_dir"

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Stop GPU monitoring
kill $MONITOR_PID 2>/dev/null || true

echo ""
echo "========================================="
echo "Experiment completed!"
echo "========================================="
echo "Processing time: $((DURATION / 60))min $((DURATION % 60))sec"

# Extract peak memory
if [ -f "gpu_memory_${EXP_KEY}.log" ]; then
    PEAK_MEM=$(grep "Memory_Used" "gpu_memory_${EXP_KEY}.log" | awk -F', ' '{print $3}' | sort -rn | head -1)
    echo "Peak memory: ${PEAK_MEM} MB"
fi

# Check if successful
if [ -f "$output_dir/pcd/combined_pcd.ply" ]; then
    echo "Status: SUCCESS"
    echo "Point cloud: $output_dir/pcd/combined_pcd.ply"
else
    echo "Status: FAILED - Check logs"
fi
