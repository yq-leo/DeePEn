export CUDA_VISIBLE_DEVICES=1,2,3

model1=openchat/openchat-3.5-0106
model2=internlm/internlm2_5-7b-chat

python src/transfer_matrix/cal_and_save_transfer_matrix.py mat ${model1} ${model2}
