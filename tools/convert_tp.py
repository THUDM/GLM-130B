import os
import sys
import torch
import argparse
import glob

from typing import *

sys.path.append(".")

SEQUENTIAL_LAYERS = [
    "input_layernorm.weight",
    "input_layernorm.bias",
    "attention.dense.bias",
    "post_attention_layernorm.weight",
    "post_attention_layernorm.bias",
    "mlp.dense_4h_to_h.bias",
    "attention.rotary_emb.inv_freq",
    "final_layernorm.weight",
    "final_layernorm.bias",
]

GLU_LAYERS = [
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
]

QUANTIZED_LAYERS = [
    "attention.dense.weight",
    "attention.query_key_value.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight",
]

LAYER_CONCAT_DIM = {"attention.dense.weight": 1, "mlp.dense_4h_to_h.weight": 1}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", default=None, type=str, help="Input SAT checkpoint folder")
    parser.add_argument("--output-folder", default=None, type=str, help="Output SAT checkpoint folder")
    parser.add_argument("--target-tp", default=4, type=int, help="Target TP degree")
    parser.add_argument("--quantization-bit-width", default=None, type=int, help="Quantization bit width")

    args = parser.parse_args()
    if args.quantization_bit_width is not None:
        assert args.quantization_bit_width in [4, 8]

    return args


def merge_weights(
    key: str,
    sd_list: List[Dict],
    tp_index: int,
    original_tp: int,
    target_tp: int,
    cat_dim: int,
    is_glu: bool,
    quantization_bit_width: Optional[int],
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if original_tp >= target_tp:
        if is_glu:
            if original_tp > target_tp:
                num_part = original_tp // target_tp
                assert len(sd_list) == num_part
                part1, part2 = [], []
                for i in range(len(sd_list)):
                    chunks = torch.chunk(sd_list[i][key], 2, dim=cat_dim)
                    part1.append(chunks[0])
                    part2.append(chunks[1])
                merged_sd = torch.cat(part1 + part2, dim=cat_dim)
            else:
                merged_sd = sd_list[0][key]
        else:
            merged_sd = torch.cat([sd[key] for sd in sd_list], dim=cat_dim)
    else:
        assert len(sd_list) == 1
        num_part = target_tp // original_tp
        if is_glu:
            offset = tp_index % num_part
            chunks = torch.chunk(sd_list[0][key], num_part * 2, dim=cat_dim)
            merged_sd = torch.cat([chunks[offset], chunks[num_part + offset]], dim=cat_dim)
        else:
            # without clone, torch will save entire tensor
            merged_sd = torch.chunk(sd_list[0][key], num_part, dim=cat_dim)[tp_index % num_part].clone()

    if quantization_bit_width is not None:
        from kernels import compress_int4_weight

        weight = merged_sd.cuda()
        weight_scale = (weight.abs().max(dim=-1).values / ((2 ** (quantization_bit_width - 1)) - 1)).half()
        weight = torch.round(weight / weight_scale[:, None]).to(torch.int8)
        if quantization_bit_width == 4:
            weight = compress_int4_weight(weight)
        return weight.cpu(), weight_scale.cpu()

    return merged_sd


def create_checkpoint(
    sd_list: List[Dict], tp_index: int, original_tp: int, target_tp: int, quantization_bit_width: Optional[int]
) -> Dict:
    new_sd = {}
    for key in sd_list[0].keys():
        name = ".".join(key.split(".")[3 if key.startswith("transformer.layers") else 1 :])
        if name in SEQUENTIAL_LAYERS:
            new_sd[key] = sd_list[0][key]
        else:
            new_sd[key] = merge_weights(
                key,
                sd_list,
                tp_index=tp_index,
                original_tp=original_tp,
                target_tp=target_tp,
                cat_dim=LAYER_CONCAT_DIM.get(name, 0),
                is_glu=name in GLU_LAYERS,
                quantization_bit_width=quantization_bit_width if name in QUANTIZED_LAYERS else None,
            )
            if quantization_bit_width is not None and name in QUANTIZED_LAYERS:
                new_sd[key], new_sd[f"{key}_scale"] = new_sd[key]
    new_sd = {"module": new_sd}
    return new_sd


def main(args):
    iteration = open(os.path.join(args.input_folder, "latest"), "r").read()
    original_tp = len(glob.glob(os.path.join(args.input_folder, iteration, "mp_rank_*_model_states.pt")))
    print(f"Iteration {iteration} from {args.input_folder} to {args.output_folder}")
    os.makedirs(args.output_folder, exist_ok=True)
    with open(os.path.join(args.output_folder, "latest"), "w") as file:
        file.write(str(iteration))
    os.makedirs(os.path.join(args.output_folder, iteration), exist_ok=True)

    for i in range(0, args.target_tp):
        save_path = os.path.join(args.output_folder, iteration, f"mp_rank_{i:02}_model_states.pt")
        print(f"Processing {save_path}")
        num_parts = original_tp // args.target_tp
        sd_list = [
            torch.load(
                os.path.join(args.input_folder, iteration, f"mp_rank_{j:02}_model_states.pt"), map_location="cpu"
            )["module"]
            for j in (
                range(i * num_parts, (i + 1) * num_parts)
                if args.target_tp <= original_tp
                else [i // (args.target_tp // original_tp)]
            )
        ]
        torch.save(create_checkpoint(sd_list, i, original_tp, args.target_tp, args.quantization_bit_width), save_path)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
