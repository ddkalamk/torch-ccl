

# Resnet50

The model is from : https://gitlab.devtools.intel.com/pytorch-ats/ats_integration/tree/master/example/imagenet

We used real data for inference there. And the dataset are from ace cluster.
```
(base) huang@mlt:~/dataset/dataset$ du -sh
146G	.
(base) huang@mlt:~/dataset/dataset$ ls train | wc -l
1001
(base) huang@mlt:~/dataset/dataset$ ls val | wc -l
1001

```

## Inference on SCYL
```
# for fp32
$ python main.py -a resnet50 -e --pretrained --sycl 0 ~/dataset/dataset

# for fp16
$ python main.py -a resnet50 -e --pretrained --fp16 1 --sycl 0 ~/dataset/dataset

or run
$ sh run.sh
```

## Inference on CPU
```
$ python main.py -a resnet50 -e --pretrained ~/dataset/dataset
```

## Train
```
# Train on cpu:
$ python main.py -a resnet50 -b 1 ~/dataset/dataset
# Train on sycl:
$ python main.py -a resnet50 -b 1 --sycl 0 ~/dataset/dataset
```