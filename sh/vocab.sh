export CUDA_VISIBLE_DEVICES=0,1,2,6

model1=meta-llama/Llama-2-13b-hf
model2=mistralai/Mistral-7B-Instruct-v0.1

python utils/get_common_vocab.py mat ${model1} ${model2}