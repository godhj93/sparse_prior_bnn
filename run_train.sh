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
#BASE_DIR="/home/hj/Projects/sparse_prior_bnn/runs/cifar10/densenet30/20250811/dnn/model_densenet30_type_dnn_data_cifar10_moped_False_std_0.001_scale_BS_prior_type_None_date_20250811-164704/"
BASE_DIR="/home/hj/Projects/sparse_prior_bnn/runs/cifar10/resnet20/20250813/dnn/resnet20_dnn_cifar10_False_0.001_BS_None_20250813-144010"
OOD='tinyimagenet'

SCRIPT="python3 train_with_good_prior.py --type uni --ood ${OOD} --model ${MODEL} --epochs ${EPOCHS} --data ${DATASET} --optimizer ${OPTIMIZER} --momentum ${MOMENTUM} --weight_decay ${WEIGHT_DECAY} --std ${STD} --prior_type ${PRIOR_TYPE} --weight"

ITERS=(10 20 30 40 50)
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
CUDA_VISIBLE_DEVICES=0 python3 train.py --type uni --model ${MODEL} --ood ${OOD} --epochs ${EPOCHS} --data ${DATASET} --optimizer ${OPTIMIZER} --momentum ${MOMENTUM} --weight_decay ${WEIGHT_DECAY} --std 1 --prior_type ${PRIOR_TYPE} &

echo "Laplace Model from SCRATCH"
PRIOR_TYPE="laplace"
CUDA_VISIBLE_DEVICES=0 python3 train.py --type uni --model ${MODEL} --ood ${OOD} --epochs ${EPOCHS} --data ${DATASET} --optimizer ${OPTIMIZER} --momentum ${MOMENTUM} --weight_decay ${WEIGHT_DECAY} --std 1 --prior_type ${PRIOR_TYPE} &
# ===============================
# Wait for All Background Jobs
# ===============================
wait

echo "✅ All jobs finished."
