#!/bin/bash

datasets=../datasets/resnet50

# for fp32
inference_with_bs256_fp32(){
    echo -e "--------------------------------------------"
    echo -e "Inference with batchsize = 128"
    echo -e "--------------------------------------------"
    python main.py -a resnet50 -e --pretrained --sycl 0 $datasets 2>&1 | tee ../results/resnet50/RN50_256.log
}

inference_with_bs1_fp32(){
    echo -e "--------------------------------------------"
    echo -e "Inference with batchsize = 128"
    echo -e "--------------------------------------------"
    python main.py -a resnet50 -e -b 1 --pretrained --sycl 0 $datasets 2>&1 | tee ../results/resnet50/RN50_1.log
}

# for fp16
inference_with_bs1_bf16(){
    echo -e "--------------------------------------------"
    echo -e "Inference with batchsize = 128"
    echo -e "--------------------------------------------"
    python main.py -a resnet50 -e --pretrained --fp16 1 --sycl 0 $datasets 2>&1 | tee ../results/resnet50/RN50_1.log
}

main(){
    # echo -e "--------------------------------------------"
    echo -e "============================================"
    echo -e "\t RN50"
    echo -e "============================================"
    inference_with_bs256_fp32
    inference_with_bs1_fp32
    # inference_with_bs1_bf16
}

main
