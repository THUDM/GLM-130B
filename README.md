<img src="resources/7D6433A42D189E2E6FBC62BE066BCE91.png">

<p align="center">
   🌐 <a href="http://keg.cs.tsinghua.edu.cn/glm-130b/posts/glm-130b/" target="_blank">Blog</a> • ⏬ <a href="https://docs.google.com/forms/d/e/1FAIpQLSehr5Dh_i3TwACmFFi8QEgIVNYGmSPwV0GueIcsUev0NEfUug/viewform" target="_blank">Download Model</a> • 🪧 <a href="https://huggingface.co/spaces/THUDM/GLM-130B" target="_blank">Demo</a> • ✉️ <a href="mailto:glm-130b@googlegroups.com">Email</a> • 📃 <a href="https://arxiv.org/abs/2210.02414" target="_blank">Paper</a><br>
</p>

<p align="center">
   💬 <a href="https://groups.google.com/g/glm-130b-forum" target="_blank">Google Group</a> (Updates) or <a href="https://github.com/THUDM/GLM-130B/blob/main/resources/WechatGroup.jpeg" target="_blank">Wechat Group</a> or <a href="https://join.slack.com/t/glm-130b/shared_invite/zt-1f2ih11xy-EAuDComTAr~XVB3MywE9Cg" target="_blank">Slack channel</a> (Discussions)
</p>

# GLM-130B: An Open Bilingual Pre-Trained Model

GLM-130B is an open bilingual (English & Chinese) bidirectional dense model with 130 billion parameters, pre-trained using the algorithm of [General Language Model (GLM)](https://aclanthology.org/2022.acl-long.26). It is designed to support inference tasks with the 130B parameters on **a single A100 (40G * 8)** or **V100 (32G * 8) server**. With INT4 quantization, the  hardware requirements can further be reduced to **a single server with 4 * RTX 3090 (24G)** with **almost no performance degradation**. As of July 3rd, 2022, GLM-130B has been trained on over 400 billion text tokens (200B each for Chinese and English) and it has the following unique features:
 
- **Bilingual:** supports both English and Chinese. 
- **Performance (EN):** better than GPT-3 175B (+4.0%), OPT-175B (+5.5%), and BLOOM-176B (+13.0%) on LAMBADA and slightly better than GPT-3 175B (+0.9%) on MMLU.
- **Performance (CN):** significantly better than ERNIE TITAN 3.0 260B on 7 zero-shot CLUE datasets (+24.26%) and 5 zero-shot FewCLUE datasets (+12.75%). 
- **Fast Inference:** supports fast inference on both [SAT](https://github.com/THUDM/SwissArmyTransformer) and [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) (up to 2.5X faster) with a single A100 server.
- **Reproducibility:** all results (30+ tasks) can be easily reproduced with open-sourced code and model checkpoints.
- **Cross-Platform:** supports training and inference on NVIDIA, Hygon DCU, Ascend 910, and Sunway (Will be released soon).

This repository mainly focus on the evaluation of GLM-130B, the training part can be found at [this repo](https://github.com/THUDM/LargeScale). If you find our work and our open-sourced efforts useful, ⭐️ to encourage our following development! :)

## News

- **[2022.10.06]** Our [paper](http://arxiv.org/abs/2210.02414) for GLM-130B is out!
- **[2022.08.24]** We are proud to publish the quantized version for GLM-130B.  While preserving the activation precision as FP16, the model weights can be quantized to as low as **INT4 with almost no degradation of performance**, further reducing the hardware requirements of the GLM-130B to **a single server with 4 * RTX 3090 (24G)**! See [Quantization of GLM-130B](docs/quantization.md) for details.

For smaller models, please find [monolingual GLMs](https://github.com/THUDM/GLM) (English: 10B/2B/515M/410M/335M/110M, Chinese: 10B/335M) and an [1B multilingual GLM](https://github.com/THUDM/Multilingual-GLM) (104 languages).

## Getting Started

### Environment Setup

#### Hardware

| **Hardware**    | **GPU Memory** | **Quantization** | **Weight Offload** |
| --------------- | -------------- | ---------------- | ------------------ |
| 8 * A100        | 40 GB          | No               | No                 |
| 8 * V100        | 32 GB          | No               | Yes (BMInf)        |
| 8 * V100        | 32 GB          | INT8             | No                 |
| 8 * RTX 3090    | 24 GB          | INT8             | No                 |
| 4 * RTX 3090    | 24 GB          | INT4             | No                 |
| 8 * RTX 2080 Ti | 11 GB          | INT4             | No        |

It is recommended to use the an A100 (40G * 8) server, as all GLM-130B evaluation results (~30 tasks) reported can be easily reproduced with a single A100 server in about half a day. With INT8/INT4 quantization, efficient inference on **a single server with 4 * RTX 3090 (24G)** is possible, see [Quantization of GLM-130B](docs/quantization.md) for details. Combining quantization and weight offloading techniques, GLM-130B can also be inferenced on servers with even smaller GPU memory, see [Low-Resource Inference](docs/low-resource-inference.md) for details.

#### Software

The GLM-130B code is built on the top of [SAT](https://github.com/THUDM/SwissArmyTransformer). We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage your environment and installing additional dependencies via `pip install -r requirements.txt`. Here are the recommended environment configurations:

- Python 3.9+ / CUDA 11+ / PyTorch 1.10+ / DeepSpeed 0.6+ / Apex (**installation with CUDA and C++ extensions is required, see [here](https://github.com/NVIDIA/apex/#linux)**)
- SwissArmyTransformer>=0.2.11 is required for quantization

#### Model weights

Download the GLM-130B’s model checkpoint from [here](https://docs.google.com/forms/d/e/1FAIpQLSehr5Dh_i3TwACmFFi8QEgIVNYGmSPwV0GueIcsUev0NEfUug/viewform?usp=sf_link), make sure all 60 chunks are downloaded completely, then use the following command to merge them into a single archive file and extract it:

```bash
cat glm-130b-sat.tar.part_* > glm-130b-sat.tar
tar xvf glm-130b-sat.tar
```

Set `CHECKPOINT_PATH` in `configs/model_glm_130b.sh` to the path of the extracted folder. Since the checkpoint file is up to 260G, it is recommended to use the SSD or RAM disk to reduce the checkpoint loading time. Since the checkpoint we distribute is in 8-way tensor parallel, a conversion scripts is also provided if you need to change the tensor parallel dimension.

```bash
python tools/convert_tp.py \
    --input-folder <SRC_CKPT_PATH>  \
    --output-folder <DST_CKPT_PATH> \
    --target-tp <TARGET_TP>
```

### Left-To-Right Generation / Blank Filling

```bash
bash scripts/generate.sh --input-source interactive
```

You can also specify an input file by `--input-source input.txt`.

GLM-130B uses two different mask tokens: `[MASK]` for short blank filling and `[gMASK]` for left-to-right long text generation. When the input does not contain any MASK token, `[gMASK]` will be automatically appended to the end of the text.

<details>
<summary><b>Left-To-Right Generation Examples</b></summary>
    
#### Example 1
Input: Who is the greatest artist? The greatest artist is [gMASK]

Output: Who is the greatest artist? The greatest artist is not the one who paints the most beautiful picture, or writes the best poem, or plays the best symphony. The greatest artist is the one whose work has the greatest impact on the lives of others. The Greatest Artist is the artist who inspires us to be our best selves. The Great Artist is not the artist whose work is flawless. The great artist is not one who is perfect. The true great artist has the ability to make us feel something. The artist who makes us feel is the great artist. The art of the great artists is to make you feel something that you have never felt before. The greatness of an artist is in the ability of their work to touch the lives and hearts of others, and to make them feel something they have never experienced before. This is the true greatness of the artist. 2. The Significance of the Great Artist The significance of the greatest artist cannot be overstated. The significance and impact of the art of a great artist cannot even be measured. The impact of a single work of art can change the course of history. A single work can change a person’s life. A work of great art can make you laugh, make you cry, make your heart race, or make you think. A great work of artistic expression has the power to change the way you look at the world. The work of a truly great artist can change your life forever. 3. The Impact of the Greatest Artist The greatest artists have the ability and the power of their art to change people’s lives. They have the power and the ability, through their art, to make people feel something new. The power of the work of the truly great artists cannot be measured, but it can be felt. The greater the impact of an artists work, the greater the power it has to change lives. 4. The Influence of the Art of the Artist The art and the work created by the greatest artists has the potential to change a life. The influence of the artists work cannot be quantified, but is felt by those who experience it. The truly great work has an ability to touch people’ lives in a way that no other work can.
  
#### Example 2 (Chinese)
Input: 问题：冬天，中国哪座城市最适合避寒？问题描述：能推荐一些国内适合冬天避寒的城市吗？回答用户：旅游爱好者 回答： [gMASK]
  
Output: 问题:冬天,中国哪座城市最适合避寒?问题描述:能推荐一些国内适合冬天避寒的城市吗?回答用户:旅游爱好者 回答:避寒的话,建议去海南三亚。 三亚位于海南岛的最南端,是中国最南部的热带滨海旅游城市,中国海滨城市,是中国空气质量最好的城市、全国最长寿地区(平均寿命80岁)。三亚市别称鹿城,又被称为"东方夏威夷",位居中国四大一线旅游城市"三威杭厦"之首,拥有全岛最美丽的海滨风光。三亚拥有被无数城市嫉妒的清新空气,柔和海滩。在沙滩悠闲散步、沐浴傍晚温和阳光,在海边玩耍,在雨林里面呼吸健康,欣赏自然奇观,一切都是那么令人享受。 气候 三亚地处热带,属热带季风气候区域,终年气温高,寒暑变化不大,年平均气温25.5度。三亚最热的月份是6月,平均气温为28.5°C,极端最高气温为35.7°C ,三亚没有明显的冬季,而且冬季更是到海滨旅游的最好时候,所以三亚是全年全天候避寒、消暑、度假、旅游的好地方。不过,三亚最旺的旅游时间是从10月份到农历春节,这个时候的人特别多,房间也不好订,建议最好避开高峰。三亚的三个旅游高峰期是"五一"、"十一"、"春节",这三个节日期间的房价都会上浮几倍,如果你选择了这三个假期来三亚的话要提前一个星期定房和咨询。
</details>

<details>
<summary><b>Blank Filling Examples</b></summary>
  
#### Example 1
Input: Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai.
  
Output: Ng is an adjunct professor at Stanford University (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai.<br/>

#### Example 2 (Chinese)
Input: 凯旋门位于意大利米兰市古城堡旁。1807年为纪念[MASK]而建，门高25米，顶上矗立两武士青铜古兵车铸像。
  
Output: 凯旋门位于意大利米兰市古城堡旁。1807年为纪念拿破仑胜利而建,门高25米,顶上矗立两武士青铜古兵车铸像。</code></pre>
</details>

<details>
<summary><b>Arguments Useful in Generation</b></summary>

- `--input-source [path] or "interactive"` The input file's path. It can also be "interactive", which will launch a CLI.
- `—-output-path [path]` The folder containing the results.
- `—-out-seq-length [int]` The maximum sequence length for generation (including context).
- `—-min-gen-length [int]` The minimum generation length for each MASK.
- `—-sampling-strategy "BaseStrategy" or "BeamSearchStrategy"`. The sampling strategy used.
  - For BeamSearchStrategy:
     - `—-num-beams [int]` The number of beams.
     - `—-length-penalty [float]` The maximum sequence length for generation (including context).
     - `—-no-repeat-ngram-size [int]` Prohibit repeated n-gram generation.
     - `—-print-all-beam` Print the generated results for all beams.
  - For BaseStrategy:
     - `—-top-k [int]` Top k sampling.
     - `—-top-p [float]` Top p sampling.
     - `—-temperature [float]` The sampling temperature.
</details>

### Evaluation

We use the YAML file to define tasks. Specifically, you can add multiple tasks or folders at a time for evaluation, and the evaluation script will automatically collect all YAML files under those folders recursively.

```
bash scripts/evaluate.sh task1.yaml task2.yaml dir1 dir2 ...
```

Download our evaluation dataset [here](https://cloud.tsinghua.edu.cn/f/9257ee84045644b8ac06/), and set `DATA_PATH` in `scripts/evaluate.sh` to your local dataset directory. The task folder contains the YAML files for 30+ tasks we evaluated for GLM-130B. Take the [CoLA](https://nyu-mll.github.io/CoLA/) task for example, run `bash scripts/evaluate.sh tasks/bloom/glue_cola.yaml`, which outputs an accuracy of ~65% for the best prompt and ~57% for the median.

<details>
<summary>Expected Output</summary>
  
```plain
MultiChoiceTaskConfig(name='glue_cola', type=<TaskType.MULTICHOICE: 'mul'>, path='/thudm/LargeScale/data/zeroshot/bloom/glue_cola', module=None, metrics=['Accuracy'], use_task_mask=False, use_multitask_encoding=False, unidirectional=False, max_seq_length=2048, file_pattern={'validation': '**/validation.jsonl'}, micro_batch_size=8)
Evaluating task glue_cola:
  Evaluating group validation:
      Finish Following_sentence_acceptable/mul/validation.jsonl, Accuracy = 42.665
      Finish Make_sense_yes_no/mul/validation.jsonl, Accuracy = 56.951
      Finish Previous_sentence_acceptable/mul/validation.jsonl, Accuracy = 65.197
      Finish editing/mul/validation.jsonl, Accuracy = 57.622
      Finish is_this_correct/mul/validation.jsonl, Accuracy = 65.197
Evaluation results of task glue_cola:
  Group validation Accuracy: max = 65.197, median = 57.622, average = 57.526
Finish task glue_cola in 101.2s. 
```
</details>

Multi-node evaluation can be configured by setting `HOST_FILE_PATH`(required by the [DeepSpeed lanucher](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node)) in `scripts/evaluate_multiple_node.sh`. Set `DATA_PATH` in `scripts/evaluate_multiple_node.sh` and run the following command to evaluate all the tasks in `./task` directory.

```
bash scripts/evaluate_multiple_node.sh ./tasks
```

See [Evaluate Your Own Tasks](docs/evaluate-your-own-tasks.md) for details on how to add new tasks.

### 2.5X faster Inference using FasterTransformer

By adapting the GLM-130B model to [FasterTransfomer](https://github.com/NVIDIA/FasterTransformer), a highly optimized transformer model library by NVIDIA, we can reach up to 2.5X speedup on generation, see [Inference with FasterTransformer](docs/inference-with-fastertransformer.md) for details.




<details>
<summary><b>Acknowledgement</b></summary>

<br/>
This project is supported by the National Science Foundation for Distinguished Young Scholars (No. 61825602). 

### Lead Contributors
[Aohan Zeng (Tsinghua KEG)](https://github.com/Sengxian), [Xiao Liu (Tsinghua KEG)](https://github.com/xiao9905)

### Contributors
#### Tsinghua KEG---the Knowledge Engineering Group at Tsinghua
Zhengxiao Du, Ming Ding, Qinkai Zheng, Hanyu Lai, Zihan Wang, Zhuoyi Yang, Jifan Yu, Xiaohan Zhang, Wendi Zheng, Xiao Xia, Yifan Xu, Weng Lam Tam, Yuxiao Dong, Jie Tang

#### Tsinghua PACMAN---the Parallel Architecture & Compiler technology of Mobile, Accelerated, and Networked systems Group at Tsinghua
Zixuan Ma, Jiaao He, Zhenbo Sun, Jidong Zhai, Wenguang Chen

#### Tsinghua NLP (BMInf)---the Natural Language Processing Group at Tsinghua
Guoyang Zeng, Xu Han, Weilin Zhao, Zhiyuan Liu
   
#### Zhipu.AI---an AI startup that aims to teach machines to think like humans
Yufei Xue, Shan Wang, Jiecai Shan, Haohan Jiang, Zhengang Guo, Peng Zhang

### Computation Sponsor
Zhipu.AI

### Project Leader
[Jie Tang (Tsinghua KEG & BAAI)](http://keg.cs.tsinghua.edu.cn/jietang/)

</details>

## License

This repository is licensed under the [Apache-2.0 license](LICENSE). The use of GLM-130B model weights is subject to the [Model License](MODEL_LICENSE).

## Citation

If you find our work useful, please consider citing GLM-130B:

```
@article{zeng2022glm130b,
  title={GLM-130B: An Open Bilingual Pre-trained Model},
  author={Zeng, Aohan and Liu, Xiao and Du, Zhengxiao and Wang, Zihan and Lai, Hanyu and Ding, Ming and Yang, Zhuoyi and Xu, Yifan and Zheng, Wendi and Xia, Xiao and Tam, Weng Lam and Ma, Zixuan and Xue, Yufei and Zhai, Jidong and Chen, Wenguang and Zhang, Peng and Dong, Yuxiao and Tang, Jie},
  journal={arXiv preprint arXiv:2210.02414},
  year={2022}
}
```

You may also consider GLM's original work in your reference:

```
@inproceedings{du2022glm,
  title={GLM: General Language Model Pretraining with Autoregressive Blank Infilling},
  author={Du, Zhengxiao and Qian, Yujie and Liu, Xiao and Ding, Ming and Qiu, Jiezhong and Yang, Zhilin and Tang, Jie},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={320--335},
  year={2022}
}
```
