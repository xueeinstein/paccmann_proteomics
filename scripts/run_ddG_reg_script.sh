TASK_NAME=$1
export MODEL_NAME=../trained_models/exp2_roberta_2seq
export DATA_DIR=/mnt/xueyang/Code/PaHelix/apps/s2f_proteomics/downstream_datasets
export CACHE_DIR=../data/fine_tuning/ddG
export OUTPUT_DIR=../trained_models/finetuning/
export BATCH_SIZE=4
export MAX_LENGTH=512
export NUM_EPOCHS=100
export SAVE_STEPS=750
export SEED=1

python ../paccmann_proteomics/run_ddG_prediction.py \
--model_name_or_path $MODEL_NAME \
--task_name $TASK_NAME \
--continue_from_checkpoint \
--data_dir $DATA_DIR \
--cache_dir $CACHE_DIR \
--output_dir $OUTPUT_DIR/$TASK_NAME \
--max_seq_length $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--learning_rate 1e-5 \
--logging_dir $OUTPUT_DIR/$TASK_NAME \
--per_device_train_batch_size $BATCH_SIZE \
--seed $SEED \
--overwrite_output_dir \
--do_train \
--do_eval
