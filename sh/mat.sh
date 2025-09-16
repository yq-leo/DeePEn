export CUDA_VISIBLE_DEVICES=0,1,2,6

model1=internlm/internlm2_5-7b-chat
model2=Qwen/Qwen3-4B-Instruct-2507

python src/transfer_matrix/cal_and_save_transfer_matrix.py mat ${model1} ${model2}