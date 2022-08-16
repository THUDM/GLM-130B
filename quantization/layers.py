import torch
from torch.nn.parameter import Parameter

from SwissArmyTransformer.mpu import copy_to_model_parallel_region
from SwissArmyTransformer.mpu import gather_from_model_parallel_region
from SwissArmyTransformer.mpu import reduce_from_model_parallel_region
from SwissArmyTransformer.mpu import scatter_to_model_parallel_region
from SwissArmyTransformer.mpu import ColumnParallelLinear, RowParallelLinear

from .functional import W8A16Linear


class QuantizedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, bit_width=8, weight=None, *args, **kwargs):
        super(QuantizedColumnParallelLinear, self).__init__(*args, **kwargs)
        self.bit_width = bit_width

        shape = self.weight.shape
        del self.weight

        if weight is None:
            self.weight = torch.empty(shape[0] * bit_width // 8, shape[1], dtype=torch.int8, device=kwargs["device"])
            self.weight_scale = torch.empty(shape[0], dtype=kwargs["params_dtype"], device=kwargs["device"])
        else:
            self.weight_scale = (weight.abs().max(dim=-1).values / ((2 ** (bit_width - 1)) - 1)).half()
            self.weight = torch.round(weight / self.weight_scale[:, None]).to(torch.int8)

        self.weight = Parameter(self.weight, requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale, requires_grad=False)

    def forward(self, input_):
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = W8A16Linear.apply(input_parallel, self.weight, self.weight_scale)
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output


class QuantizedRowParallelLinear(RowParallelLinear):
    def __init__(self, bit_width=8, weight=None, *args, **kwargs):
        super(QuantizedRowParallelLinear, self).__init__(*args, **kwargs)
        self.bit_width = bit_width

        shape = self.weight.shape
        del self.weight

        if weight is None:
            self.weight = torch.empty(shape[0] * bit_width // 8, shape[1], dtype=torch.int8, device=kwargs["device"])
            self.weight_scale = torch.empty(shape[0], dtype=kwargs["params_dtype"], device=kwargs["device"])
        else:
            self.weight_scale = (weight.abs().max(dim=-1).values / ((2 ** (bit_width - 1)) - 1)).half()
            self.weight = torch.round(weight / self.weight_scale[:, None]).to(torch.int8)

        self.weight = Parameter(self.weight, requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale, requires_grad=False)

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = W8A16Linear.apply(input_parallel, self.weight, self.weight_scale)
        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output
