import torch


class W8A16Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, quant_w: torch.Tensor, scale_w: torch.Tensor):
        ctx.inp_shape = inp.size()
        ctx.weight_shape = quant_w.size()
        out_features = quant_w.size(0)
        inp = inp.contiguous().view(-1, quant_w.size(1))
        weight = quant_w.to(torch.half) * scale_w[:, None]
        output = inp.mm(weight.t())
        ctx.save_for_backward(inp, quant_w, scale_w)
        return output.view(*(ctx.inp_shape[:-1] + (out_features,)))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inp, quant_w, scale_w = ctx.saved_tensors
        weight = quant_w.to(torch.half) * scale_w[:, None]
        grad_output = grad_output.contiguous().view(-1, weight.size(0))
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(inp)
        return grad_input.view(ctx.inp_shape), grad_weight.view(ctx.weight_shape), None
