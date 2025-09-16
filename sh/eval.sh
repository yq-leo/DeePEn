task=MMLU
run_mode=test
models=InternLM7b+qwen4b

python utils/evaluate/EM_dir_test.py res/${task}/${run_mode}/${models}/vanilla
# python utils/evaluate/EM_dir_test.py res/${task}/${run_mode}/${models}/tas
# python utils/evaluate/EM_dir_test.py res/${task}/${run_mode}/${models}/tas2
# python utils/evaluate/EM_dir_test.py res/${task}/${run_mode}/${models}/tas2+mas2
# python utils/evaluate/EM_dir_test.py res/${task}/${run_mode}/${models}/tas3+mas2
