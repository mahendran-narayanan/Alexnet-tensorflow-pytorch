# Alexnet-tensorflow-pytorch

Alexnet model can be created in Tensorflow, Pytorch packages.

```
usage: main.py [-h] [--model {tf,torch}]

Create alexnet model in Tensorflow or Pytorch package

optional arguments:
  -h, --help          show this help message and exit
  --model {tf,torch}  Model created on Tensorflow, Pytorch (default: Tensorflow)
```

To create model in Tensorflow:

```
python3 main.py --model tf
```

To create model in Pytorch:

```
python3 main.py --model torch
```

Paper link: [Alexnet paper](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)