from generation import BeamSearchStrategy


class BeamSearchStrategyForLAMBADA(BeamSearchStrategy):
    def __init__(self, *args, banned_prefix=[], **kwargs):
        super().__init__(*args, **kwargs)
        self.banned_prefix = banned_prefix

    def forward(self, logits, tokens, mems):
        batch_size, vocab_size = logits.shape
        logits = logits.float()
        for prefix in self.banned_prefix:
            if self.length_generated == len(prefix) - 1:
                if len(prefix) == 1:
                    logits[..., prefix[0]] = -65504
                else:
                    for i in range(batch_size):
                        if tokens[i, -(len(prefix) - 1) :].tolist() == prefix[:-1]:
                            logits[i, prefix[-1]] = -65504
        return super().forward(logits, tokens, mems)
