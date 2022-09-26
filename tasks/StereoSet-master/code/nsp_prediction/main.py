import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from argparse import ArgumentParser
import dataset
from torch.utils.data import DataLoader
import sys
sys.path.append("../models/")
import models
import transformers
import numpy as np
from sklearn.metrics import accuracy_score 

# We use Apex to speed up training on FP16.
# It is also needed to train any GPT2-[medium,large,xl].
try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

def parse_args():
    args = ArgumentParser()
    args.add_argument("--batch-size", default=16, type=int)
    args.add_argument("--no-cuda", default=False, action="store_true")
    args.add_argument("--dataset", default="out", type=str)
    args.add_argument("--model", default="RobertaModel", choices=["XLNetModel", "RobertaModel", "BertModel", \
            "GPT2Model"])
    args.add_argument("--pretrained-class", default="roberta-base") 
    args.add_argument("--epochs", default=1, type=int)
    args.add_argument("--data-frac", default=0.1, type=float)
    args.add_argument("--skip-frac", default=0.003, type=float, help="Amount of training data to skip from the beginning.")
    args.add_argument("--max-seq-length", default=256, type=int)
    args.add_argument("--core-lr", default=5e-6, type=float)
    args.add_argument("--head-lr", default=1e-3, type=float)
    args.add_argument("--weight-decay", default=1e-2, type=float)
    args.add_argument("--tokenizer", default="RobertaTokenizer")
    args.add_argument("--saved-model", default=None)
    args.add_argument("--test", default=False, action="store_true")
    args.add_argument("--fp16", default=False, action="store_true")
    args.add_argument("--opt", default="O0", choices=["O0", "O1", "O2", "O3"]) 
    args.add_argument("--accumulation-steps", type=int, default=1)
    args.add_argument("--local_rank", type=int, default=None) 
    args.add_argument("--world-size", type=int, default=None) 
    return args.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    torch.manual_seed(5)
    if args.local_rank is not None:
        local_rank = args.local_rank
        print(f"Using GPU ID {local_rank}")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        dist.init_process_group(backend="nccl", init_method="env://")

        # An alternative method to perform distributed training.
        # dist.init_process_group(backend="nccl", init_method="file:///temp/parallel_comm", \
                # world_size=args.world_size, rank=local_rank)
    else:
        local_rank = None
        device = "cpu" if args.no_cuda else "cuda"

    # Create Model
    if local_rank is not None:
        model = models.ModelNSP(args.pretrained_class).cuda(local_rank)
    else:
        model = models.ModelNSP(args.pretrained_class).to(device)

    model.core_model.output_past = False

    if args.test:
        model.eval()
    else:
        model.train()

    print(f"Number of parameters: {count_parameters(model):,}")
    print(f"Gradient Accumulation Steps: {args.accumulation_steps}")

    tokenizer = getattr(transformers, args.tokenizer).from_pretrained(args.pretrained_class)
    if "gpt2" in args.tokenizer.lower():
        # this enables us to do batched training, GPT2 wasn't trained with a padding token.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        model.core_model.resize_token_embeddings(len(tokenizer))

    criterion = nn.CrossEntropyLoss()

    # the pretrained model has been fairly optimized, while the NSP head has been randomly initialized.
    # using different learning rates helps speed up training.
    specific_learning_rates = [{"params": model.core_model.parameters(), "lr": args.core_lr, "correct_bias": False}, {"params": model.nsp_head.parameters(), "lr": args.head_lr, "correct_bias": False}]
    optimizer = transformers.AdamW(specific_learning_rates, lr=args.core_lr, correct_bias=False)

    fp16 = args.fp16
    if fp16: 
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt, keep_batchnorm_fp32=True)

    if local_rank is not None:
      print(f"Device is set to {device}!")
    else:
      print("Let's use", torch.cuda.device_count(), "GPUs!")

    if local_rank is not None:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    else:
        model = nn.DataParallel(model)

    print("Passed model distribution stage!")
    if args.saved_model:
        sd = torch.load(args.saved_model, map_location=device)
        model.load_state_dict(sd)
        model.to(device)

    # Create Dataset
    data = dataset.NextSentenceDataset(args.dataset, tokenizer, data_frac=args.data_frac, max_seq_length=args.max_seq_length, test=args.test, skip_frac=args.skip_frac)
    if local_rank is not None:
        sampler = torch.utils.data.distributed.DistributedSampler(data, \
	    num_replicas=args.world_size, rank=local_rank)  
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=shuffle, num_workers=0, \
	sampler=sampler, pin_memory=True)
    test_scores = []
    accumulation_steps = args.accumulation_steps
    num_training_steps = len(dataloader) // accumulation_steps * args.epochs

    print(f"Total Training Steps: {num_training_steps}")
    scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=250, num_training_steps=num_training_steps)
    # Also try
    # scheduler = ReduceLROnPlateau(optimizer, "max", patience=10, verbose=True)

    # Train
    for epoch in range(args.epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        ticks = 0.0
        number_of_batches = len(dataloader)
        for train_batch_num, example in enumerate(dataloader):
            input_ids = torch.stack(example[0], dim=0).transpose(0, 1)
            token_type_ids = torch.stack(example[1], dim=0).transpose(0, 1)
            attention_mask = torch.stack(example[2], dim=0).transpose(0, 1)
            labels = example[3]

            if local_rank is not None:
                input_ids = input_ids.cuda(non_blocking=True)
                token_type_ids = token_type_ids.cuda(non_blocking=True)
                attention_mask = attention_mask.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            else:
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()

            output, loss = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
            output_probs = output.softmax(dim=-1)
            predictions = torch.argmax(output_probs, dim=1)

            loss = loss.mean(dim=0)
            loss = loss / accumulation_steps
            running_loss += loss.item() 

            accuracy = accuracy_score(predictions.detach().cpu().numpy(), labels.detach().cpu().numpy())
            if args.test:
                test_scores.append(accuracy)
            running_accuracy += accuracy 
            ticks += 1.0

            if not args.test:
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                if (train_batch_num) % accumulation_steps == 0:
                    if fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0) 
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

            if ((train_batch_num * args.batch_size) % 500==0 and train_batch_num>0):
                for param_group in optimizer.param_groups:
                    print("LR:", param_group['lr'])
                acc = (running_accuracy / ticks) 
                loss = (running_loss / ticks) * accumulation_steps
                progress = train_batch_num / number_of_batches 
                print(f"[Epoch {epoch+1}: {progress*100:.2f}%] Accuracy: {acc}, Loss: {loss}")
                running_loss = 0.0
                running_accuracy = 0.0
                ticks = 0.0

    if args.test:
        print(f"Final test accuracy: {np.mean(test_scores)}")

    if not args.test and (local_rank==0 or local_rank is None):
        save_path = f"trained_models/ft_{args.model}_{args.pretrained_class}_{args.core_lr}_{args.head_lr}.pth" 
        print(f"Saving model to {save_path}")
        torch.save(model.state_dict(), save_path)


if __name__=="__main__":
    args = parse_args()
    main(args)
