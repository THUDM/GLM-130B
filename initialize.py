import argparse
import torch
import time

from quantization import quantize

from SwissArmyTransformer import get_args, get_tokenizer
from SwissArmyTransformer.arguments import initialize_distributed
from SwissArmyTransformer.training import load_checkpoint
from SwissArmyTransformer.model import GLM130B


def add_bminf_args(parser):
    """Arguments for BMInf"""
    group = parser.add_argument_group("BMInf")

    group.add_argument("--bminf", action="store_true", help="Use BMInf to support low resource evaluation")
    group.add_argument("--bminf-memory-limit", type=int, default=20, help="Max memory for model per GPU (in GB)")
    return parser


def add_quantization_args(parser):
    group = parser.add_argument_group("Quantization")

    group.add_argument("--quantization-bit-width", type=int, default=None)
    group.add_argument("--from-quantized-checkpoint", action="store_true", help="Loading from a quantized checkpoint")


def initialize(extra_args_provider):
    parser = argparse.ArgumentParser(add_help=False)
    add_bminf_args(parser)
    add_quantization_args(parser)
    GLM130B.add_model_specific_args(parser)
    extra_args_provider(parser)
    known, args_list = parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    args.do_train = False
    initialize_distributed(args)
    return args


def initialize_model_and_tokenizer(args):
    tokenizer = get_tokenizer(args)

    # Initialize model
    model = GLM130B(args).half()

    if args.from_quantized_checkpoint:
        assert args.quantization_bit_width is not None
        # Quantize model before moving to GPU
        model = quantize(model, args.quantization_bit_width)

    # Load checkpoint
    torch.distributed.barrier()
    start = time.time()
    load_checkpoint(model, args)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(f"> Checkpoint loaded in {time.time() - start:.1f}s")

    if args.quantization_bit_width is not None and not args.from_quantized_checkpoint:
        # Quantize model before moving to GPU
        model = quantize(model, args.quantization_bit_width)

    if args.bminf:
        import bminf

        if torch.distributed.get_rank() == 0:
            print(f"> BMInf activated, memory limit: {args.bminf_memory_limit} GB")
        with torch.cuda.device(args.device):
            model = bminf.wrapper(model, quantization=False, memory_limit=args.bminf_memory_limit << 30)
    else:
        model = model.to(args.device)

    torch.cuda.empty_cache()
    model.eval()

    # generate rotary embedding cache
    original_parallel_output = model.transformer.parallel_output
    model.transformer.parallel_output = True
    with torch.no_grad():
        _, *_ = model(
            torch.ones(1, args.max_sequence_length, device=torch.cuda.current_device(), dtype=torch.int64),
            torch.arange(args.max_sequence_length, device=torch.cuda.current_device(), dtype=torch.int64).view(1, -1),
            torch.randn(
                1,
                1,
                args.max_sequence_length,
                args.max_sequence_length,
                device=torch.cuda.current_device(),
            )
            < 0.5,
        )
    model.transformer.parallel_output = original_parallel_output
    torch.distributed.barrier()

    return model, tokenizer
