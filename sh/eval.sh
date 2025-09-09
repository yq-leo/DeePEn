task=NQ
run_mode=test
models=OpenChat+InternLM7b

python utils/evaluate/EM_dir_test.py res/${task}/${run_mode}/${models}/vanilla
python utils/evaluate/EM_dir_test.py res/${task}/${run_mode}/${models}/tas
# python utils/evaluate/EM_dir_test.py res/${task}/${run_mode}/${models}/tas2
python utils/evaluate/EM_dir_test.py res/${task}/${run_mode}/${models}/tas2+mas2
