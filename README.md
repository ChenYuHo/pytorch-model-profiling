## Cloning repo

```bash
git clone https://github.com/ChenYuHo/pytorch-model-profiling.git
cd pytorch-model-profiling
git submodule update --init # --depth 1
```

## Building and activating the Conda environment (Assuming mamba is correctly installed)

The script should be run from the project root directory as follows. 

```bash
./bin/create-conda-env.sh
conda activate ./env
```

If you have multiple GPUs, select one to use for profiling, otherwise errors may occur
```bash
export NVIDIA_VISIBLE_DEVICES=0
```


## Run torchvision CNN profilings

```bash
cd src
# assuming conda environment already activated
python pytorch_module_hooks_profiler.py MODEL
```

check
```bash
python pytorch_module_hooks_profiler.py -h
```
for more options.


## Run BERT profilings (fine-tuning on SQuAD)
```bash
cd src/DeepLearningExamples/PyTorch/LanguageModeling/BERT
# Follow README.md to prepare checkpoint and training data in $base_dir such that
# $base_dir/pytorch/bert_uncased.pt
# $base_dir/squad/v1.1/train-v1.1.json
# $base_dir/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt
# are available. Update run_profiling.sh accordingly.
./scscripts/run_profiling.sh
```
Reference results can be found at:

A100: https://github.com/ChenYuHo/DeepLearningExamples/tree/bert-profiling/PyTorch/LanguageModeling/BERT#fine-tuning-nvidia-dgx-a100-8x-a100-80gb

V100: https://github.com/ChenYuHo/DeepLearningExamples/tree/bert-profiling/PyTorch/LanguageModeling/BERT#fine-tuning-nvidia-dgx-1-with-32g

## Result parsing

`get_model_size_and_fp_bp_median.py` extracts model size, median of forward pass, backward pass, and weight update times of every layer. The optional BUCKET_SIZE_MB groups small layers into one layer, so every "bucket" is less than BUCKET_SIZE_MB (best-effort)
```bash
python get_model_size_and_fp_bp_median.py [TRACE_JSON_PATH] [BUCKET_SIZE_MB]

```
