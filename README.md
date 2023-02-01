# NanoGPT-Tensorflow

This is a rewrite of Andrej Karpathy's nanoGPT repo in Tensorflow.
I did it for learning purposes and hopefully in can help out those who would prefer to use Tensorflow instead of PyTorch.

I encourage all to take a look at his repo and accompanying video.

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [# Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## Install

Dependencies:

- [pytorch](https://pytorch.org/) <3
- [numpy](https://numpy.org/install/) <3
- `pip install transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
- `pip install datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
- `pip install tiktoken` for OpenAI's fast BPE code <3
- `pip install wandb` for optional logging <3
- `pip install tqdm`

## Usage

To render a dataset we first tokenize some documents into one simple long 1D array of token indices. E.g. for OpenWebText run:

```
$ cd data/openwebtext
$ python prepare.py

```

To download and tokenize the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. This will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. The training script currently by default tries to reproduce the smallest GPT-2 released by OpenAI, i.e. the 124M version of GPT-2. We can train as follows on a single device, though I encourage you to read the code and see all of the settings and paths up top in the file:

```
$ python train.py

```

If you do not have GPU also add `--device=cpu --compile=False`, though you'd have to also adjust the default network size to be much much smaller (see "i only have a macbook" section below). To train using PyTorch Distributed Data Parallel (DDP) run the script with torchrun. For example to train on a node with 4 GPUs run:

```
$ torchrun --standalone --nproc_per_node=4 train.py
```

If you're in a cluster environment and are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```
Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py

```

It is a good idea to benchmark your interconnect (e.g. iperf3). In particular, if you don't have Infiniband then also prepend `NCCL_IB_DISABLE=1` to the above launches. Your multinode training will work, but most likely _crawl_.

By default checkpoints are periodically written to the `--out_dir` (`./out` by default). Once we have one, we can sample from the model:

```
$ python sample.py

```

## [](https://github.com/karpathy/nanoGPT#baselines)Baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```
$ python train.py eval_gpt2
$ python train.py eval_gpt2_medium
$ python train.py eval_gpt2_large
$ python train.py eval_gpt2_xl

```

and observe the following losses on train and val:

| Model       | Params | Train Loss | Val Loss |
| ----------- | ------ | ---------- | -------- |
| gpt2        | 124M   | 3.11       | 3.12     |
| gpt2-medium | 350M   | 2.85       | 2.84     |
| gpt2-large  | 774M   | 2.66       | 2.67     |
| gpt2-xl     | 1558M  | 2.56       | 2.54     |

However, we have to note that GPT-2 was trained on (closed, never released) WebText, while OpenWebText is just a best-effort open reproduction of this dataset. This means there is a dataset domain gap. Indeed, taking the GPT-2 (124M) checkpoint and finetuning on OWT directly for a while reaches loss down to ~2.9. This then becomes the more appropriate baseline w.r.t. reproduction.
