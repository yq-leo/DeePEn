export CUDA_VISIBLE_DEVICES=0,2,4,6

model1=meta-llama/Llama-2-13b-hf
model2=openchat/openchat_3.5
model3=mistralai/Mistral-7B-Instruct-v0.1

python src/transfer_matrix/cal_and_save_transfer_matrix.py mat ${model1} ${model2} ${model3}
