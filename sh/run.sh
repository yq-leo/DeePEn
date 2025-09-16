export CUDA_VISIBLE_DEVICES=6,4,7,2

task=MMLU
rm=test
models=qwen4b+InternLM7b+OpenChat
mode=tas2+mas2

res_path=./res/${task}/${rm}/${models}/${mode}
log_path=./log/${task}/${rm}/${models}/${mode}
mkdir -vp ${res_path}
mkdir -vp ${log_path}

nohup python src/main_many_ensemble_llama_series_local_matrix.py --config confs/${task}/${models}.json -lpm based_on_probility_transfer_logits_fp32_processor -d0 cuda:0 -d1 cuda:0 -d2 cuda:0 -dp cuda:1  -rsd ${res_path} -rm ${rm} -lr 0.15 -em ${mode} > ${log_path}/run.log 2>&1 &
