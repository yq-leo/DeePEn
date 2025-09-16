export CUDA_VISIBLE_DEVICES=6,4,7,2

model1=internlm/internlm2_5-7b-chat
model2=Qwen/Qwen3-4B-Instruct-2507

python src/transfer_matrix/cal_and_save_transfer_matrix.py mat ${model1} ${model2}
