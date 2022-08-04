# Low-resource Inference with BMInf

GLM-130B is trained with 4-way tensor parallel and 8-way pipeline parallel for efficiency. Then the checkpoint is converted into a 8-way tensor parallel one in order to inference the model in a single node. GLM-130B has 130 billion parameters in FP16 precision, a total of 260G of GPU memory is required to store model weights. The DGX-A100 server has 8 A100s and provides an amount of 320G of GPU memory (640G for 80G A100 version)  so it suits GLM-130B well. 

However, a server with 8 * 32G V100 only provides an amount of 256G of GPU memory, which indicates that the full loading of model weights is not possible. Fortunately, with the swap-in-and-out feature between CPU and GPU memory provided by the [BMInf](https://github.com/OpenBMB/BMInf) library, GLM-130B can still run on servers with a smaller amount of GPU memory. After joint debugging with the BMInf team, we achieved a resonable evaluation efficiency on DGX-1 servers with 8 * 32G V100 by carefully overlapping computation and communication, see the [benchmark section](#benchmark) for details.

We have integrated BMInf into our codebase, just install BMInf via `pip install bminf`, and change the model configuration file from `configs/model_glm_130b.sh` to `configs/model_glm_130b_v100.sh` in your launch shell script. The default BMInf config is for V100 servers, you can also adjust the maximum memory the model weights can occupy on one GPU by setting `--bminf-memory-limit` according to your GPU memory in the model config file.

## Benchmark

### Evaluation

- CoLA task on the validation set
- Micro Batch Size = 30
- BMInf: 25GB model weights in GPU memory limit by: `--bminf-memory-limit 25`

|                | Peak GPU Memory | Time   |
| -------------- | ---------- | ------ |
| A100-SAT       | 40.3 G     | 74.6 s |
| V100-SAT       | OOM        | OOM    |
| V100-SAT-BMInf | 32.3 G     | 196.0 s |

The `micro-batch-size` config in task YAML files is configured according to the maximum utilization of the DGX-A100 server. If you encounter an OOM error on the V100 server, please adjust the `micro-batch-size` appropriately.

### Text generation

In text generation, due to the small amount of calculation per model forward (usually <10 tokens/forward using beam search strategy), the communication between the CPU and GPU memory becomes the bottleneck. With the help of the BMInf team, we did an in-depth profile on our V100 server. Given a 25GB model weight limit per GPU, a total of 13 layers need to be copied from CPU to GPU for a single forward, each layer will take about 75ms on IO, indicating that the real IO speed between CPU and GPU is `260GB / 70 / 8 / 75ms = 6.19GB/s`. Our V100 server uses PCI-E 3.0 and two V100s share a switch, so the theoretical bandwidth for each GPU is 8GB/s, close to our profiling results. A server with PCI-E 4.0 will greatly reduce the IO time. Even that, long text generation tokens can still take several minutes so **we do not recommend using V100 servers in text generation scenario**. For this, we are working on INT8 quantization so that GLM-130B can even fit a single RTX-3090 server (24G * 8).

