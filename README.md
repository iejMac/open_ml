# OpenML

Generic template for ML projects

## Usage

```
pip install open_ml
```

```python
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

image = preprocess(Image.open("CLIP.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
```

### Pretrained models

We offer a simple model interface to instantiate both pre-trained and untrained models.
To see which pretrained models are available, use the following code snippet.
More details about our pretrained models are available [here](docs/PRETRAINED.md).

```python
>>> import open_clip
>>> open_clip.list_pretrained()
```

### Loading models

Models can be loaded with `open_clip.create_model_and_transforms`, as shown in the example below. The model name and corresponding `pretrained` keys are compatible with the outputs of `open_clip.list_pretrained()`. 

The `pretrained` argument also accepts local paths, for example `/path/to/my/b32.pt`.
You can also load checkpoints from huggingface this way. To do so, download the `open_clip_pytorch_model.bin` file (for example, [https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/tree/main](https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/blob/main/open_clip_pytorch_model.bin)), and use `pretrained=/path/to/open_clip_pytorch_model.bin`.

```python
# pretrained also accepts local paths
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k') 
```

## Training OpenML

### Install

We advise you first create a virtual environment with:

```bash
python3 -m venv .env
source .env/bin/activate
pip install -U pip
```

You can then install openclip for training with `pip install 'open_clip_torch[training]'`.

### Data

To prepare the data (for this example implementation) you can run:
```bash
python -m open_ml.data.download --data-dir ./data
```

#### Development

If you want to make changes to contribute code, you can clone openclip then run `make install` in openclip folder (after creating a virtualenv)

Install pip PyTorch as per https://pytorch.org/get-started/locally/

You may run `make install-training` to install training deps

#### Testing

Test can be run with `make install-test` then `make test`

`python -m pytest -x -s -v tests -k "training"` to run a specific test

Running regression tests against a specific git revision or tag:
1. Generate testing data
    ```sh
    python tests/util_test.py --model RN50 RN101 --save_model_list models.txt --git_revision 9d31b2ec4df6d8228f370ff20c8267ec6ba39383
    ```
    **_WARNING_: This will invoke git and modify your working tree, but will reset it to the current state after data has been generated! \
    Don't modify your working tree while test data is being generated this way.**

2. Run regression tests
    ```sh
    OPEN_CLIP_TEST_REG_MODELS=models.txt python -m pytest -x -s -v -m regression_test
    ```

### Sample single-process running code:

```bash
python -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/path/to/train_data.csv"  \
    --val-data="/path/to/validation_data.csv"  \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=/path/to/imagenet/root/val/ \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=30 \
    --workers=8 \
    --model RN50
```

### Multi-GPU and Beyond

This code has been battle tested up to 1024 A100s and offers a variety of solutions
for distributed training. We include native support for SLURM clusters.

#### Single-Node

We make use of `torchrun` to launch distributed jobs. The following launches a
a job on a node of 4 GPUs:

```bash
cd open_clip/src
torchrun --nproc_per_node 4 -m training.main \
    --train-data '/data/cc12m/cc12m-train-{0000..2175}.tar' \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --batch-size 320 \
    --precision amp \
    --workers 4 \
    --imagenet-val /data/imagenet/validation/
```

#### Multi-Node

The same script above works, so long as users include information about the number
of nodes and host node.

```bash
cd open_clip/src
torchrun --nproc_per_node=4 \
    --rdzv_endpoint=$HOSTE_NODE_ADDR \
    -m training.main \
    --train-data '/data/cc12m/cc12m-train-{0000..2175}.tar' \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --batch-size 320 \
    --precision amp \
    --workers 4 \
    --imagenet-val /data/imagenet/validation/
```

#### SLURM

This is likely the easiest solution to utilize. The following script was used to
train our largest models:

```bash
#!/bin/bash -x
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=open_clip
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition PARTITION_NAME

eval "$(/path/to/conda/bin/conda shell.bash hook)" # init conda
conda activate open_clip
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd /shared/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"
srun --cpu_bind=v --accel-bind=gn python -u src/training/main.py \
    --save-frequency 1 \
    --report-to tensorboard \
    --train-data="/data/LAION-400M/{00000..41455}.tar" \
    --warmup 2000 \
    --batch-size=256 \
    --epochs=32 \
    --workers=8 \
    --model ViT-B-32 \
    --name "ViT-B-32-Vanilla" \
    --seed 0 \
    --local-loss \
    --gather-with-grad
```

### Resuming from a checkpoint:

```bash
python -m training.main \
    --train-data="/path/to/train_data.csv" \
    --val-data="/path/to/validation_data.csv"  \
    --resume /path/to/checkpoints/epoch_K.pt
```

### Logging

For tensorboard logging, run:
```bash
tensorboard --logdir=logs/tensorboard/ --port=7777
```

For wandb logging, we recommend looking at the `step` variable instead of `Step`, since the later was not properly set in earlier versions of this codebase.
For older runs with models trained before https://github.com/mlfoundations/open_clip/pull/613, the `Step` variable should be ignored.
For newer runs, after that PR, the two variables are the same.

## Evaluation / Zero-Shot

### Evaluating local checkpoint:

```bash
python -m training.main \
    --val-data="/path/to/validation_data.csv"  \
    --model RN101 \
    --pretrained /path/to/checkpoints/epoch_K.pt
```

### Evaluating hosted pretrained checkpoint on ImageNet zero-shot prediction:

```bash
python -m training.main \
    --imagenet-val /path/to/imagenet/validation \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32
```

### Gradient accumulation

To simulate larger batches use `--accum-freq k`. If per gpu batch size, `--batch-size`, is `m`, then the effective batch size will be `k * m * num_gpus`.

When increasing `--accum-freq` from its default of 1, samples/s will remain approximately constant (batch size will double, as will time-per-batch). It is recommended to use other features to reduce batch size such as `--grad-checkpointing --local-loss --gather-with-grad` before increasing `--accum-freq`. `--accum-freq` can be used in addition to these features.

Instead of 1 forward pass per example, there are now 2 forward passes per-example. However, the first is done with `torch.no_grad`.

There is some additional GPU memory required --- the features and data from all `m` batches are stored in memory.

There are also `m` loss computations instead of the usual 1.

For more information see Cui et al. (https://arxiv.org/abs/2112.09331) or Pham et al. (https://arxiv.org/abs/2111.10050).

### Support for remote loading/training

It is always possible to resume directly from a remote file, e.g., a file in an s3 bucket. Just set `--resume s3://<path-to-checkpoint> `.
This will work with any filesystem supported by `fsspec`.

It is also possible to train `open_clip` models while continuously backing up to s3. This can help to avoid slow local file systems.

Say that your node has a local ssd `/scratch`, an s3 bucket `s3://<path-to-bucket>`.

In that case, set `--logs /scratch` and `--remote-sync s3://<path-to-bucket>`. Then, a background process will sync `/scratch/<run-name>` to `s3://<path-to-bucket>/<run-name>`. After syncing, the background process will sleep for `--remote-sync-frequency` seconds, which defaults to 5 minutes.

There is also experimental support for syncing to other remote file systems, not just s3. To do so, specify `--remote-sync-protocol fsspec`. However, this is currently very slow and not recommended.

Also, to optionally avoid saving too many checkpoints locally when using these features, you can use `--delete-previous-checkpoint` which deletes the previous checkpoint after saving a new one.

Note: if you are using this feature with `--resume latest`, there are a few warnings. First, use with `--save-most-recent` is not supported. Second, only `s3` is supported. Finally, since the sync happens in the background, it is possible that the most recent checkpoint may not be finished syncing to the remote.

### Pushing Models to Hugging Face Hub

The module `open_clip.push_to_hf_hub` includes helpers for pushing models /w weights and config to the HF Hub.

The tool can be run from command line, ex:
`python -m open_clip.push_to_hf_hub --model convnext_large_d_320 --pretrained /train/checkpoints/epoch_12.pt --repo-id laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft`


## Acknowledgments

We gratefully acknowledge the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) for funding this part of work by providing computing time through the John von Neumann Institute for Computing (NIC) on the GCS Supercomputer JUWELS Booster at Jülich Supercomputing Centre (JSC).
