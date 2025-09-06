
# DeePEn (Ensemble Learning for Heterogeneous Large Language Models with Deep Parallel Collaboration)
<div align=center><img src="figures/overview.png" width="80%" /></div>

Source code for paper [Ensemble Learning for Heterogeneous Large Language Models with Deep Parallel Collaboration](https://arxiv.org/html/2404.12715).

![](./figures/Method.jpg "DeePEn")

In this paper, we propose a *training-free* method **DeePEn** to fuse the probability distributions ouput by heterogeneous LLMs, which have different vocabularies (e.g., LLaMA and Mistral). At each decoding step, DeePEn determines the next token according to the fused distribution, improving the performance across all experimental benchmarks (MMLU, ARC-C, GSM8K, PIQA, TriviaQA, and NQ).
![](./figures/performance.png "performance")
![](./figures/Main_Experiment.jpeg "performance")

Ensemble learning between Mixtral-8x7b and LLaMA2-70B:
<div align=center><img src="figures/Ensemble_Dense_and_Sparse.png" width="50%" /></div>

Ensemble learning between LLM and multilingual translator NLLB:
<div align=center><img src="figures/Ensemble_LLM_and_Expert.png" width="50%" /></div>


We exemplify the usage of our code with the ensemble learning of LLaMA2-13B, Mistral-7B, InternLM-20B, and TigerBot-13B on the NQ dataset.

## Usage

### Step-1: Construct Relative Representation Matrix

In sh/mat.sh file, modify the path of model1 and model2 (and model3, ...), then run

```bash
sh sh/mat.sh
```

### Step-2: Configuration

In conf files, Fill in the following fields:

- `model_path`: Paths to the model
- `probability_transfer_matrix_path`: Directory path to the constructed relative representation matrix
- `file_path`: Paths to the example prompt and the validation and test datasets, which are also provided in this repository (./datasets)

### Step-3: Inference

Modify sh/run.sh file to specify task, run_mode(rm), models, and mode(choose from vanilla, tas, tas2, tas2+mas2).

```bash
sh sh/run.sh
```

### Step-4: Evaluation

Modify sh/eval.sh file to specify task, run_mode(rm), models, and mode(choose from vanilla, tas, tas2, tas2+mas2).


```bash
sh sh/eval.sh
```

## Requirements
- torch==2.1.2
- transformers==4.40.0

## Citation

```
@misc{huang2024enabling,
      title={Enabling Ensemble Learning for Heterogeneous Large Language Models with Deep Parallel Collaboration}, 
      author={Yichong Huang and Xiaocheng Feng and Baohang Li and Yang Xiang and Hui Wang and Bing Qin and Ting Liu},
      year={2024},
      eprint={2404.12715},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
