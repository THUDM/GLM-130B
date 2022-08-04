# GLM-130B 训练日志

## 模型信息

- 130B：70 layers，12288 hidden size，32768 ffn hidden size, 150000 vocab size
   - MP = 4, PP = 8
- GLM + Rotary Positional Embedding + GeGLU + DeepNorm
- FP32 softmax with QKV scaling（no PB-Relax）
- Shrink embedding gradient with $\alpha=0.1$
- Global batch size: 4224

## 环境版本

- PyTorch 1.11 / CUDA 11.3
- LargeScale@400893da37bb5cbe22c29e41c02a052369cc72ce
- DeepSpeed 0.6.1
- apex@master

## 测速

- 96 nodes, BSZ=176 * 24=4224
   - glm-130B-2022.05.05-19:34:16：134TFLOPS, 88.5s/iter, 48samples/s,
- 96 nodes, BSZ=256 * 24=6144
   - glm-130B-2022.05.05-19:43:13：141TFLOPS, 122.5s/iter, 50samples/s

## 2022-05-06 04:00 开始训练

- glm-130B-2022.05.05-19:53:15

## 2022-05-07 20:14 节点故障

坏掉 n30041, n30157 两个点，更改保存间隔为 100step，从 4000 step 开始训练

- glm-130B-2022.05.07-13:44:59

## 2022-05-10 00:00 提升 alpha

加入 `--shrink-embedding-gradient-steps 6000 500` 从 6000 step 开始训练

- glm-130B-2022.05.09-16:02:04

## 2022-05-11 12:13 节点故障

坏掉 n30115 节点，从 7300 step 开始训练

- glm-130B-2022.05.11-05:55:32

## 2022-05-20 00:03 节点故障

坏掉 n30066 节点，从 15400 step 开始训练

- glm-130B-2022.05.19-19:56:19

再换一批节点，从 15600 step 开始训练

- glm-130B-2022.05.20-01:58:57

## 2022-05-21 12:40 换节点

训练效率一直只有 127T 左右，怀疑之前加入的 n30076 存在问题，踢出后从 16600 step 开始训练，似乎不解决问题。

## 2022-05-22 19:27 节点故障

n30126 失联

- glm-130B-2022.05.22-14:15:41

## 2022-05-26 04:30 节点故障

n30039 掉卡

- glm-130B-2022.05.25-22:23:12


## 2022-05-28 11:50 更换中英多任务数据（废除）

从 22800 开始训练，换中英多任务数据

- glm-130B-2022.05.28-03:52:26
- events.out.tfevents.1653709957.9droa42ltcad5-0.1858.0（移除）

## 2022-05-28 16:50 更换英文多任务数据（废除）

换新的多任务数据 22900 左右出现 nan，挂掉训练，检查发现中文多任务数据噪声极大，从 22800 换成平衡后的 t0 原始数据开始训练

- glm-130B-2022.05.28-09:18:12
- events.out.tfevents.1653729502.9droa42ltcad5-0.5648.0（移除）

## 2022-05-28 20:50 加入 warmup（废除）

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/C850748B-92A4-4F9F-932F-AD22330895D6_2/E8MboG8vrTTb2N51FRhkb6wsB4eyrD77USmM992obQgz/Image.png)

换上平衡后且不泄漏的 t0 原始数据开始训练仍然有问题，推测是平衡后一些任务占比变大，其实等价于加入新任务的情况，加入参数 `--warmup-samples-after-loading 2112000` warmup 500 步从 22800 开始训练

- glm-130B-2022.05.28-12:57:24
- events.out.tfevents.1653742654.9droa42ltcad5-0.7942.0（移除）

## 2022-05-29 01:30 再次爆炸，换纯文本（废除）

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/028DE014-00FE-4521-BEEB-EF3F61BB8DA1_2/mgYybTR1OLgPkBysqMiUgGYNyIg8OQnf1yXI66grBeMz/Image.png)

- warmup 以后还是炸了，分析可能是 distribution 变动仍然太过剧烈，先换纯文本 + reshuffle 尝试训练，从 22800 加载
- glm-130B-2022.05.28-18:05:33
- events.out.tfevents.1653761143.9droa42ltcad5-0.9744.0（废除）
- global_step23200_text
+ 配置文件

## 2022-05-29 逐渐修改数据分布（废除）

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/E2BC463F-E519-461E-B1B0-99551DA940BE_2/0ZqN22TLyqRTvqOy6JNLeixEy4TarDJEF7DOvdh3saIz/Image.png)

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/9C7AC4B3-59AB-471A-872E-41CCBAE7E90D_2/0rpEmyAOcIkLyDGR2R4RQiBeUwbWIWiaHbHcwosx6yAz/Image.png)

文本似乎能稳定，那么尝试逐渐平滑修改数据分布， 从 22800 开始，逐渐修改数据分布到 t0 平衡数据

- glm-130B-2022.05.29-05:17:06
- events.out.tfevents.1653801436.9droa42ltcad5-0.13868.0（废除）

## 2022-05-29 22:40 逐渐修改数据分布并全面 warmup

- 又挂了，分析可能是换新分布学习率也需要 warmup

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/F5532A86-3AAC-4CCE-AC9B-A976B7736D7F_2/M4JZx5GYzNPuysPHXrn0R5Oo54rBhDwQxdErkOpFOhEz/Image.png)

- 从 22800 开始训练，数据和 lr 都 warmup 2000 步，shrink embbeding graident 从 0.2 warmup 6000 步到 1
- glm-130B-2022.05.29-17:35:45

## 2022-05-30 14:00 挂节点

更改了一下参数配置，发现之前 shrink embedding 的步数写错了（26850 步），现在改成 6000 步。升级了一下 lr auto warmup 的逻辑，写成绝对 samples 数量。从 global_step23200 开始

我们发现这次训练卡在了数据加载，排查后发现是 Lustre 文件系统的故障，导致 2.3T 文本数据读不出来，且工程师无法修复；最终重新从移动硬盘拷贝了一次数据

- glm-130B-2022.05.31-02:18:24

## 2022.05.03 20:00 加 DeepStruct 数据

- 维持原有 transform 过程不变，但直接加入 DeepStruct 数据，从 23500 开始

## 2022-06-01 22:22 换清洗数据

之前的多任务数据 t0 和 deepsturct 各有一个任务的 target 异常，重新清洗后更换，从 24500 开始

- glm-130B-2022.06.01-14:24:33

## 2022-06-02 12:00 节点故障

- n30145 CPU 故障，从 25000 重启训练，lr 和 数据集已经 transfromer 完毕，所以配置直接去掉 warmup
- glm-130B-2022.06.02-04:35:05

## 2022-06-02 09:30 加入 multitask loss 打印

25800steps 开始，加入 multitask loss 打印

- glm-130B-2022.06.03-01:40:12

## 2022-06-02 15:00 降低学习率，加入 gpt/bert loss 打印

loss 降低比较慢，讨论可能是学习率太大了，26000steps 开始，学习率砍半

- glm-130B-2022.06.03-07:26:16

## 2022-06-06 17:00 集群维护

集群从 9 点到 5 点升级驱动，从  开始训练

- glm-130B-2022.06.06-10:00:39

PS：观察到共享文件系统读取速度显著改善，现在加载 ckpt 几乎只需要 1 分钟

## 2022-06-08 08:00 坏点

- glm-130B-2022.06.08-00:00:37

## 2022-06-09 13:30 训练卡住

23100 开始恢复

- glm-130B-2022.06.09-05:27:54

## 2022-06-12 10:00 loss 爆炸

33700 开始 loss 炸了，loss-scale 在 33710 左右突然下跌然后 loss 在 33740 左右爆炸

- tensorboard 记录：glm-130B-33700

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/C46C7CFE-1B79-491C-90FC-5A88AE90E9DF_2/7ICMyH8v6GhAgngz5bVaDKwzYjFPyk99Ax27R5w56wMz/Image.png)

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/E56BCDE0-C798-429F-81E0-1A07CCB9BC0E_2/Ig2rfKnPmLadg39Jc38UEdK90LDxlAxoH0AxmAygxzAz/Image.png)

- 从 33600 开始加载，shrink embedding gradient 1 → 0.5
- glm-130B-2022.06.12-02:20:49

## 2022-06-14 03:00 loss 爆炸

35250 loss 又炸了，和 33700 的表现几乎一样，都是完全没有征兆突然爆炸

tensorboard 记录：glm-130B-35250

- 从 35200 开始加载，shrink embedding gradient 0.5 → 0.1
- glm-130B-2022.06.14-02:28:21

## 2022-06-19 00:10 节点故障

n30085 挂了，从 39600 恢复

- glm-130B-2022.06.18-17:49:53

## 2022-06-20 09:10 loss 爆炸

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/CA344108-3B01-469C-9ABE-C41002F76484_2/oEvBST5MP0I7S4qHmQUeE7DoPCsGFSrveAOOSyitSUwz/Image.png)

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/FED0DE40-A710-4259-AE98-26BCB9568C7A_2/kH4FijsPDVJFzkbaxz7BiX0RZrul1Wrye6cE5EV8ZG0z/Image.png)

- tensorboard 记录：glm-130B-40800
- `--skip-train-iteration-range 40701-40900`
- 从 40700 开始重新加载并跳过 40701-40900 数据
- glm-130B-2022.06.20-03:36:13

## 2022-06-22 10:40 梯度 spike

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/0B7E0A0C-4B11-4F52-BF10-E6B11A533BEF_2/yb1zC07di9zux8jbAi15gpqlstGHXZyjyMBEjO0gNKUz/Image.png)

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/1C60424A-0290-4070-9327-DF9DFD135020_2/XyVoPs1yMLIuzUyrDixSYfgjc2Y2Nuor20GCz0nSPkAz/Image.png)

- grad 有点小 spike，看起来后续恢复了，但 loss 似乎遇到了比较大的波动
- `--skip-train-iteration-range 40701-40900`
- 从 42400 开始重新加载并跳过 42401-42600 数据
- glm-130B-2022.06.22-02:38:20

## 2022-06-22 21:00 梯度 spike

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/E406CC41-4180-4108-BCCF-5E727CEB8F09/1D7D801C-3226-4CB0-978C-F19B4DA46721_2/nmg9r87OFrdErZvY9xjiDIHvgPVLv39vy8ZVtGkj2H0z/Image.png)

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/E406CC41-4180-4108-BCCF-5E727CEB8F09/5F5CA3D6-AF58-4087-9806-1529D3A2EF6C_2/WSQqyBdv1rvzvNloXE6Ssql7GxMDoULU38FAQCv3778z/Image.png)

- grad 又有 spike，但是 loss-scale 没有一降到底，推测应该可以恢复
- 这几天的反复 spike，我们分析可能是后期 learning rate 降低太慢，将 min-lr 从 8e-6 调整到 4e-6
- `--min-lr 4e-6`
- 从 42700 加载开始训练
- glm-130B-2022.06.22-13:03:53

## 2022.06.26 16:00 节点故障

- 节点 NVLink Error，重启训练
- glm-130B-2022.06.26-13:13:51

## 2022.06.29 00:00 恢复 position_id

- 48100 从原先配置开始训练
- glm-130B-2022.06.29-13:53:21
