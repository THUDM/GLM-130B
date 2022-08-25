# The training notes of GLM-130B 

## Basic Information about GLM-130B

- 130B：70 layers，12288 hidden size，32768 ffn hidden size, 150000 vocab size
   - MP = 4, PP = 8
- GLM + Rotary Positional Embedding + GeGLU + DeepNorm
- FP32 softmax with QKV scaling（no PB-Relax）
- Shrink embedding gradient with $\alpha=0.1$
- Global batch size: 4224

## Environment

- PyTorch 1.11 / CUDA 11.3
- LargeScale@400893da37bb5cbe22c29e41c02a052369cc72ce
- DeepSpeed 0.6.1
- apex@master

## Speed Testing (with Different Batch Sizes)

- 96 nodes, BSZ=176 * 24=4224
   - glm-130B-2022.05.05-19:34:16：134TFLOPS, 88.5s/iter, 48samples/s,
- 96 nodes, BSZ=256 * 24=6144
   - glm-130B-2022.05.05-19:43:13：141TFLOPS, 122.5s/iter, 50samples/s

## 2022-05-06 04:00 Training starts

- glm-130B-2022.05.05-19:53:15

## 2022-05-07 20:14 Node failure

n30041, n30157 break down, changing saving interval to 100 steps (originally 500 steps, too long), restart from 4000 step

- glm-130B-2022.05.07-13:44:59

## 2022-05-10 00:00 Increase alpha for embedding shrink, as we think the original alpha is too small (originally 0.1)

add `--shrink-embedding-gradient-steps 6000 500` to warmup alpha to 1 from 6000 step within 500 steps

- glm-130B-2022.05.09-16:02:04

## 2022-05-11 12:13 Node failure

n30115 breaks down, restart from 7300 step

- glm-130B-2022.05.11-05:55:32

## 2022-05-20 00:03 Node failure

n30066 breaks down, restart from 15400 step

- glm-130B-2022.05.19-19:56:19

Switch to another node pool, and restart from 15600 step

- glm-130B-2022.05.20-01:58:57

## 2022-05-21 12:40 Replace node

Finding that the training flop is only 127T, smaller than before; suspecting that the n30076 we have replaced in has some unknown errors and kicking it out from 16600 step; nothing changes

## 2022-05-22 19:27 Node failure

n30126 loses connection

- glm-130B-2022.05.22-14:15:41

## 2022-05-26 04:30 Node failure

n30039 reports missing GPUs

- glm-130B-2022.05.25-22:23:12


## 2022-05-28 11:50 Change Multi-task Instruction Pre-training (MIP) data (abolished)

Restarts from 22800 step, change MIP data to the correct one (English & Chinese)

- glm-130B-2022.05.28-03:52:26
- events.out.tfevents.1653709957.9droa42ltcad5-0.1858.0 (abolished)

## 2022-05-28 16:50 Change MIP data

New MIP data (English & Chinese) leads to NaN loss at 22900 step; finding too much noises in Chinese multi-task data; switch to vanilla T0 training datasets

- glm-130B-2022.05.28-09:18:12
- events.out.tfevents.1653729502.9droa42ltcad5-0.5648.0（移除）

## 2022-05-28 20:50 Add warmup (abolished)

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/C850748B-92A4-4F9F-932F-AD22330895D6_2/E8MboG8vrTTb2N51FRhkb6wsB4eyrD77USmM992obQgz/Image.png)

Vanilla T0 datasets still lead to disconvergence; suspecting a changed task ratio leads to the instability; add argument `--warmup-samples-after-loading 2112000` to warmup 500 steps from 22800 step

- glm-130B-2022.05.28-12:57:24
- events.out.tfevents.1653742654.9droa42ltcad5-0.7942.0（移除）

## 2022-05-29 01:30 Disconverges again, switch to self-supervised pre-training only (abolished)

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/028DE014-00FE-4521-BEEB-EF3F61BB8DA1_2/mgYybTR1OLgPkBysqMiUgGYNyIg8OQnf1yXI66grBeMz/Image.png)

- Disconverges after warmup; suspecting that the distribution change is still too large; trying to restart using self-supervised pre-training only with data reshuffle, loading from 22800 step
- glm-130B-2022.05.28-18:05:33
- events.out.tfevents.1653761143.9droa42ltcad5-0.9744.0 (abolished)
- global_step23200_text
+ Configuration file

## 2022-05-29 Smoothing distribution shift (abolished)

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/E2BC463F-E519-461E-B1B0-99551DA940BE_2/0ZqN22TLyqRTvqOy6JNLeixEy4TarDJEF7DOvdh3saIz/Image.png)

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/9C7AC4B3-59AB-471A-872E-41CCBAE7E90D_2/0rpEmyAOcIkLyDGR2R4RQiBeUwbWIWiaHbHcwosx6yAz/Image.png)

Self-supervised pre-training only seems to be stable; trying to smooth the distribution shift via a warmed-up ratio of correct T0 data from 22800 step

- glm-130B-2022.05.29-05:17:06
- events.out.tfevents.1653801436.9droa42ltcad5-0.13868.0 (abolished)

## 2022-05-29 22:40 Smoothing data distribution shift & warmup learning rate

- Disconverges; suspecting that learning rate requires warmup in this process, too

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/F5532A86-3AAC-4CCE-AC9B-A976B7736D7F_2/M4JZx5GYzNPuysPHXrn0R5Oo54rBhDwQxdErkOpFOhEz/Image.png)

- Restart from 22800, warmup correct MIP data ratio and learning rate for 2000 steps; warmup embedding gradient shrink alpha from 0.2 to 1 by 6000 steps
- glm-130B-2022.05.29-17:35:45

## 2022-05-30 14:00 Node and file system failure

Finding the warmup steps for embedding gradient shrink to be wrong (26850 steps instead of 6000 steps); changing the warmup steps implementation (according to the absolute number of samples); restarting from global_step23200

We discover that the restart is stacked in the data loading, which turns out to be an error of the Lustre file system. The result is that we cannot read the 2.3T text corpora and the engineer cannot help to recover the data, and we have to copy data from backup disk to the file system again (which takes few days)

- glm-130B-2022.05.31-02:18:24

## 2022.05.03 20:00 Add DeepStruct data to MIP

- Keeping the original warmup process; adding DeepStruct data to MIP portion; restart from 23500 step

## 2022-06-01 22:22 Replace MIP data to a cleaner version

Finding one noisy prompt in the task data for T0 (qqp) and DeepStruct respectively; removing them and restarting from 24500 step

- glm-130B-2022.06.01-14:24:33

## 2022-06-02 12:00 Node failure

- n30145 CPU error, restarting from 25000 step; removing the warmup process as it has ended
- glm-130B-2022.06.02-04:35:05

## 2022-06-02 09:30 Start to print multitask loss

From 25800 step, we print multitask loss

- glm-130B-2022.06.03-01:40:12

## 2022-06-02 15:00 Reduce learning rate and print gpt/bert loss 

The loss decreases slowly, and we think it might be attributed to a too large learning rate; from 26000 step, we half the learning rate

- glm-130B-2022.06.03-07:26:16

## 2022-06-06 17:00 Node cluster maintenance

The node cluster needs an upgrade from 9 am to 5 am

- glm-130B-2022.06.06-10:00:39

PS: we observe a significant improvement of the file system's reading speed; only need 1 minute to load the checkpoint now

## 2022-06-08 08:00 Node failure

- glm-130B-2022.06.08-00:00:37

## 2022-06-09 13:30 Unexpected termination of the training

Restarting from 23100 step; suspecting the network communication problem

- glm-130B-2022.06.09-05:27:54

## 2022-06-12 10:00 Loss explodes

From 33700 step, the training loss explodes. The loss-scale reduces drastically around 33710 step, and the loss explodes at 33740 step

- tensorboard record：glm-130B-33700

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/C46C7CFE-1B79-491C-90FC-5A88AE90E9DF_2/7ICMyH8v6GhAgngz5bVaDKwzYjFPyk99Ax27R5w56wMz/Image.png)

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/E56BCDE0-C798-429F-81E0-1A07CCB9BC0E_2/Ig2rfKnPmLadg39Jc38UEdK90LDxlAxoH0AxmAygxzAz/Image.png)

- Restaring from 33600 step, reduce shrink embedding gradient from 1.0 to 0.5
- glm-130B-2022.06.12-02:20:49

## 2022-06-14 03:00 Loss explodes

At 35250 step, the loss explodes again; almost the same behavior as it is in 33700 step; breaking down without any signs

tensorboard record：glm-130B-35250

- Restarting from 35200 step, and shrinking embedding gradient from 0.5 to 0.1
- glm-130B-2022.06.14-02:28:21

## 2022-06-19 00:10 Node failure

n30085 breaks down, restarting from 39600 step

- glm-130B-2022.06.18-17:49:53

## 2022-06-20 09:10 Loss explodes

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/CA344108-3B01-469C-9ABE-C41002F76484_2/oEvBST5MP0I7S4qHmQUeE7DoPCsGFSrveAOOSyitSUwz/Image.png)

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/FED0DE40-A710-4259-AE98-26BCB9568C7A_2/kH4FijsPDVJFzkbaxz7BiX0RZrul1Wrye6cE5EV8ZG0z/Image.png)

- tensorboard record：glm-130B-40800
- `--skip-train-iteration-range 40701-40900`
- Restarting from 40700 step and skipping the noisy data in 40701-40900 steps
- glm-130B-2022.06.20-03:36:13

## 2022-06-22 10:40 Gradient spikes

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/0B7E0A0C-4B11-4F52-BF10-E6B11A533BEF_2/yb1zC07di9zux8jbAi15gpqlstGHXZyjyMBEjO0gNKUz/Image.png)

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/A8DAC1A6-2A03-489A-8A11-BFAFFFEE3905/1C60424A-0290-4070-9327-DF9DFD135020_2/XyVoPs1yMLIuzUyrDixSYfgjc2Y2Nuor20GCz0nSPkAz/Image.png)

- The gradient norm experiences a spike, which seems to recover automatically; but the training loss experiences a drastic change
- `--skip-train-iteration-range 40701-40900`
- Restarting from 42400 and skipping data in 42401-42600 steps
- glm-130B-2022.06.22-02:38:20

## 2022-06-22 21:00 Gradient spikes

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/E406CC41-4180-4108-BCCF-5E727CEB8F09/1D7D801C-3226-4CB0-978C-F19B4DA46721_2/nmg9r87OFrdErZvY9xjiDIHvgPVLv39vy8ZVtGkj2H0z/Image.png)

![Image.png](https://res.craft.do/user/full/97ed555f-7125-cca2-fd7d-9f1a0585132e/doc/E406CC41-4180-4108-BCCF-5E727CEB8F09/5F5CA3D6-AF58-4087-9806-1529D3A2EF6C_2/WSQqyBdv1rvzvNloXE6Ssql7GxMDoULU38FAQCv3778z/Image.png)

- The gradient norm experiences a spike again, but the loss-scale seems stable. We think it might recover automatically.
- Rethinking on the repeating gradient spikes in recent days, we speculate it might be attributed to a too-slow learning rate decay in the late stage of pre-training; reducing minimum lr from 8e-6 to 4e-6
- `--min-lr 4e-6`
- Restarting from 42700 step
- glm-130B-2022.06.22-13:03:53

## 2022.06.26 16:00 Node failure

- Unexpected NVLink Error; restarting training
- glm-130B-2022.06.26-13:13:51

## 2022.06.29 00:00 Recover position_id

- Restarting training from 48100 step; using another more consistent positional encoding (the original one has a different implementation for \[MASK\] and \[gMASK\])
- glm-130B-2022.06.29-13:53:21
