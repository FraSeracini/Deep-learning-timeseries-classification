#!/bin/bash

# ==========================
# Run all Pirate Pain Prediction models in series
# ==========================

MODELS=("LSTM" "GRU" "Transformer" "CNN1D" "MLP" "Ensemble" "GNN" "TCN" "TimesNet" "BNN" "CNN_RNN")

BATCH_SIZE=64
NUM_EPOCHS=200
LEARNING_RATE=0.001
VALID_SPLIT=0.2
EARLY_STOPPING_PATIENCE=30
K_FOLDS=7

for MODEL in "${MODELS[@]}"; do
    echo "==============================="
    echo "Running model: $MODEL"
    echo "==============================="
    python3 script.py \
        --model_type $MODEL \
        --batch_size $BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --valid_split $VALID_SPLIT \
        --early_stopping_patience $EARLY_STOPPING_PATIENCE \
        --k_folds $K_FOLDS
done

echo "All models have been trained."
