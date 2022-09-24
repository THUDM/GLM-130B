import torch
import time
from initialize import initialize, initialize_model_and_tokenizer

if __name__ == "__main__":
    args = initialize(extra_args_provider=lambda parser: None)
    model, tokenizer = initialize_model_and_tokenizer(args)

    for seq_len in [512, 1024, 2048]:
        torch.distributed.barrier()
        start = time.time()
        with torch.no_grad():
            _, *_ = model(
                torch.ones(1, seq_len, device=torch.cuda.current_device(), dtype=torch.int64),
                torch.arange(seq_len, device=torch.cuda.current_device(), dtype=torch.int64).view(1, -1),
                torch.randn(1, 1, seq_len, seq_len, device=torch.cuda.current_device()) < 0.5,
            )
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print(f"Encode {seq_len}: {(time.time() - start) * 1000:.2f} ms")
