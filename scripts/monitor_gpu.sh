#!/bin/bash

# GPU memory monitoring script
# Usage: ./scripts/monitor_gpu.sh <experiment_name>

LOG_FILE="gpu_memory_${1}.log"

echo "Timestamp, GPU, Utilization, Memory_Used, Memory_Total" > "$LOG_FILE"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    GPU_INFO=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
    echo "$TIMESTAMP, $GPU_INFO" >> "$LOG_FILE"
    sleep 1
done
EOF
