# HugeCTR + DGL

This only works on multi-gpu right now and not multi-node.

## Redis + Rocksdb setup

The instruction for setting up a local test environment can be done are here: https://gitlab-master.nvidia.com/dl/hugectr/hugectr_inference_backend/-/tree/main/samples/hierarchical_deployment

## Setup

docker run -d -t --network=host --cap-add SYS_NICE --ipc=host --gpus all --name hps-dgl -v ${redis_and_rocksdb_location}:/data -v {this_repo}:/workspace nvcr.io/nvidia/merlin/merlin-training:22.04 bash

## build
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSM="70;80" ..
make -j && make install
```
