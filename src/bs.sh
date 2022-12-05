#!/bin/bash
for model in alexnet googlenet inception_v3 resnet101 resnet152 resnet50 vgg11 vgg16 vgg19; do
    for i in {0..20}; do
        bs=$((2 ** i))
        python pytorch_module_hooks_profiler.py $model -bs $bs -nb 2 >& /dev/null
        if [[ $? -ne 0 ]]; then
            echo "$model failed at bs=$bs"
            break
        else
            echo "$model bs=$bs passed"
        fi
    done
    # get result
    bs=$((2 ** (i-1)))
    echo "obtain trace for $model bs=$bs"
    python pytorch_module_hooks_profiler.py $model -bs $bs -nb 100 --out ${model}_bs${bs}
done
