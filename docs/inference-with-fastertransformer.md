# Inference with FasterTransformer

[FasterTransformer](https://github.com/NVIDIA/FasterTransformer) provides a script and recipe to run the highly optimized transformer-based encoder and decoder component, and it is tested and maintained by NVIDIA.

We adapted the GLM-130B based on Fastertransformer for fast inference, with details in [benchmark](#benchmark) section.

## Download the Model

See [Get Model](/README.md#environment-setup).

## Recommend: Run With Docker

Use Docker to quickly build a Flask API application for GLM-130B.

### Requirements

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Build Container Image

```bash
git clone https://github.com/THUDM/FasterTransformer.git
cd FasterTransformer
bash docker/build.sh
```

### Run API With Checkpoints

Set MPSIZE to the number of gpus needed for the checkpoints, and DATA_TYPE to checkpoints precision. The checkpoint we distribute is in 8-way tensor parallel in FP16 precision, a conversion scripts is also provided if you need to change the tensor parallel dimension and the weight precision.

```bash
# Convert the checkpoint to MP=4, DATA_TYPE=INT4
python tools/convert_tp.py \
    --input-folder <SRC_CKPT_PATH>  \
    --output-folder <DST_CKPT_PATH> \
    --target-tp 8 \
    --quantization-bit-width 4 \
# Run API
docker run -it --rm --gpus all --shm-size=10g -p 5000:5000 \
           -v <DST_CKPT_PATH>/49300:/checkpoints:ro \
           -e MPSIZE=4 -e DATA_TYPE=int4 \
           ftglm:latest
```

### Test

#### Benchmark

```bash
python3 examples/pytorch/glm/glm_server_test.py
```

#### Web Demo

```bash
pip install gradio
python3 examples/pytorch/glm/glm_server_frontend_test.py
```

## Manual Configuration

### Requirements

- CMake >= 3.13 for PyTorch
- CUDA 11.0 or newer version
- NCCL 2.10 or newer version
- Python 3 is recommended because some features are not supported in python 2
- PyTorch: Verify on 1.10.1, >= 1.8.0 should work.

### Setup Using Docker

```bash
docker run -it --rm --gpus all nvcr.io/nvidia/pytorch:22.09-py3 /bin/bash
conda install -y pybind11
```

### Setup Using Conda

As another way, all the packages can be installed using conda.

> Some of our current [structure](https://github.com/THUDM/FasterTransformer/blob/main/src/fastertransformer/th_op/glm/GlmOp.h#L30) requires that `g++` and `libtorch` produce the same results, so a pre-compiled `libtorch` may only work with `g++-7` or `g++-9`. And although GLM-130B itself does not rely on openmpi, FasterTransformer requires it during the build process. We are working on these issues.

```bash
conda install -y cmake pybind11
conda install -y -c conda-forge cudatoolkit-dev cudnn
cp -r $CONDA_PREFIX/lib/libcudnn* /usr/local/cuda/lib64/
cp -r $CONDA_PREFIX/include/cudnn*.h /usr/local/cuda/include/
```

If it's hard to install cudatoolkit-dev and cudnn by conda, just install them from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads), and make sure cmake is able to find cudnn.

```bash
cp cudnn/include/cudnn*.h /usr/local/cuda/include
cp cudnn/lib/libcudnn* /usr/local/cuda/lib64
chmod a+r /usr/local/cuda/include/cudnn*.h 
chmod a+r /usr/local/cuda/lib64/libcudnn*
```

GLM-130B is trained with FP16 precision, a total of 260G of GPU memory is required to store model weights. The model is tested with 8 * 40G A100s.

### Build

Get the code and install all dependencies:

```bash
git clone https://github.com/THUDM/FasterTransformer.git
mkdir -p FasterTransformer/build
cd FasterTransformer/build
pip3 install icetk transformers
```

Note: the `xx` of `-DSM=xx` in following scripts means the compute capability of your GPU. For example, 60 (P40) or 61 (P4) or 70 (V100) or 75(T4) or 80 (A100) or 86(RTX 3090).  Default setting is including 70, 75, 80 and 86.

```bash
cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
make -j
```

### Run GLM-130B

Generate the `gemm_config.in` file.

```bash
# ./bin/gpt_gemm <batch_size> <beam_width> <max_input_len> <head_number> <size_per_head> <inter_size> <vocab_size> <data_type> <tensor_para_size>
./bin/gpt_gemm 1 1 128 96 128 49152 150528 1 8
```

Running GLM_130B in Pytorch and Flask.

```bash
bash ../examples/pytorch/glm/glm-server.sh
```

You need to check and edit this file to set arguments such as `CHECKPOINT_PATH`.

## Optimization methods

Optimization in GLM_130B are similar to optimization in GPT and GPT-J, describing in the [FasterTransformer/gpt_guide.md](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md). Meanwhile, some of the operators are differ from GPT, such as the implementation of RotaryEmbedding, and the use of GeGLU, so we add them additionally into FasterTransformer.

## Benchmark

- Hardware: DGX-A100(8 * 40G)

## Encode

| **Sequence Len**   | 512    | 1024   | 2048   |
| ---------- | ------ | ------ | ------ |
| Megatron   | 145 ms | 250 ms | 453 ms |
| FasterTransformer | 120 ms | 220 ms | OOM  |

## Decode

| **Sequence Len**  | 512     | 1024    | 2048     |
| ---------- | ------- | ------- | -------- |
| Megatron   | 45.21 s | 89.00 s | 179.22 s |
| FasterTransformer | 18.77 s | 39.81 s | 89.88 s  |
