#!/usr/bin/env bash
#
# run_pruned_models_parallel.sh
# Description: Run pruned and original models on GPUs using round-robin parallelism.

# ===============================
# Configurable Variables
# ===============================
DATASET="cifar10"
MODEL="resnet20"
OPTIMIZER="sgd"
MOMENTUM=0.9
WEIGHT_DECAY=0.0
STD=1e-3
LR=1e-1
EPOCHS=90
BATCH_SIZE=128
PRIOR_TYPE="normal"
BASE_DIR="/home/hj/Projects/sparse_prior_bnn/runs/cifar10/resnet20/20250725/dnn/"
SCRIPT="python3 train_with_good_prior.py --type uni --model ${MODEL} --epochs ${EPOCHS} --data ${DATASET} --optimizer ${OPTIMIZER} --momentum ${MOMENTUM} --weight_decay ${WEIGHT_DECAY} --std ${STD} --prior_type ${PRIOR_TYPE} --weight"

ITERS=(10 20 30)
GPUS=(0 1)

# ===============================
# Launch Pruned Models
# ===============================
echo "Launching ${#ITERS[@]} pruned jobs on GPUs: ${GPUS[*]}"

for idx in "${!ITERS[@]}"; do
  iter=${ITERS[$idx]}
  dev=${GPUS[$(( idx % ${#GPUS[@]} ))]}

  echo "[Iter $iter] → GPU $dev"
  CUDA_VISIBLE_DEVICES=$dev \
    $SCRIPT "${BASE_DIR}/pruned_model_iter_${iter}.pth" &
  # run in background
  sleep 0.2  # small delay to avoid overload

done

# ===============================
# Launch Original Models
# ===============================
echo "[Original A] → GPU 0"
CUDA_VISIBLE_DEVICES=0 $SCRIPT "${BASE_DIR}/original_model.pth" &

echo "[Original B] → GPU 1"
CUDA_VISIBLE_DEVICES=1 $SCRIPT "${BASE_DIR}/original_model.pth" --MOPED &

echo "Baseline Model N(0, 1) from SCRATCH"
CUDA_VISIBLE_DEVICES=0 python3 train.py --type uni --model ${MODEL} --epochs ${EPOCHS} --data ${DATASET} --optimizer ${OPTIMIZER} --momentum ${MOMENTUM} --weight_decay ${WEIGHT_DECAY} --std ${STD} --prior_type ${PRIOR_TYPE} &
# ===============================
# Wait for All Background Jobs
# ===============================
wait

echo "✅ All jobs finished."
