# Alexnet-tensorflow-pytorch

```
usage: main.py [-h] [--model {tf,torch}]

Create alexnet model in Tensorflow or Pytorch package

optional arguments:
  -h, --help          show this help message and exit
  --model {tf,torch}  Model created on Tensorflow, Pytorch (default: Tensorflow)
```

To run in Tensorflow:

```
python3 main.py --model tf
```

To run in Pytorch:

```
python3 main.py --model torch
```