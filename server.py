import time
import torch
import torch.distributed as dist
import gradio as gr

from generation import BeamSearchStrategy, BaseStrategy
from initialize import initialize, initialize_model_and_tokenizer
from generate import add_generation_specific_args, fill_blanks


def generate_continually(func, raw_text):
    if not raw_text:
        return "Input should not be empty!"
    try:
        start_time = time.time()
        answer = func(raw_text)
        if torch.distributed.get_rank() == 0:
            print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
        return answer
    except (ValueError, FileNotFoundError) as e:
        print(e)
        return "Error!"


strategy = None


def main(args):
    model, tokenizer = initialize_model_and_tokenizer(args)

    end_tokens = [tokenizer.get_command("eop"), tokenizer.get_command("eos")]

    def process(raw_text):
        global strategy

        if args.with_id:
            query_id, raw_text = raw_text.split("\t")

        answers, answers_with_style, blanks = fill_blanks(raw_text, model, tokenizer, strategy, args)

        if torch.distributed.get_rank() == 0:
            print(answers)

        return answers[0]

    def predict(
        text,
        seed=1234,
        out_seq_length=200,
        min_gen_length=20,
        sampling_strategy="BaseStrategy",
        num_beams=4,
        length_penalty=0.9,
        no_repeat_ngram_size=3,
        temperature=1,
        topk=1,
        topp=1,
    ):

        global strategy

        if torch.distributed.get_rank() == 0:
            print(
                "info",
                [
                    text,
                    seed,
                    out_seq_length,
                    min_gen_length,
                    sampling_strategy,
                    num_beams,
                    length_penalty,
                    no_repeat_ngram_size,
                    temperature,
                    topk,
                    topp,
                ],
            )
            dist.broadcast_object_list(
                [
                    text,
                    seed,
                    out_seq_length,
                    min_gen_length,
                    sampling_strategy,
                    num_beams,
                    length_penalty,
                    no_repeat_ngram_size,
                    temperature,
                    topk,
                    topp,
                ],
                src=0,
            )

        args.seed = seed
        args.out_seq_length = out_seq_length
        args.min_gen_length = min_gen_length
        args.sampling_strategy = sampling_strategy
        args.num_beams = num_beams
        args.length_penalty = length_penalty
        args.no_repeat_ngram_size = no_repeat_ngram_size
        args.temperature = temperature
        args.top_k = topk
        args.top_p = topp

        if args.sampling_strategy == "BaseStrategy":
            strategy = BaseStrategy(
                batch_size=1, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, end_tokens=end_tokens
            )
        elif args.sampling_strategy == "BeamSearchStrategy":
            strategy = BeamSearchStrategy(
                batch_size=1,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                consider_end=True,
                end_tokens=end_tokens,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                min_gen_length=args.min_gen_length,
            )
        else:
            raise ValueError(f"unknown strategy {args.sampling_strategy}")

        return generate_continually(process, text)

    if torch.distributed.get_rank() == 0:
        en_fil = ["The Starry Night is an oil-on-canvas painting by [MASK] in June 1889."]
        en_gen = ["Eight planets in solar system are [gMASK]"]
        ch_fil = ["凯旋门位于意大利米兰市古城堡旁。1807年为纪念[MASK]而建，门高25米，顶上矗立两武士青铜古兵车铸像。"]
        ch_gen = ["三亚位于海南岛的最南端,是中国最南部的热带滨海旅游城市 [gMASK]"]
        en_to_ch = ["Pencil in Chinese is [MASK]."]
        ch_to_en = ['"我思故我在"的英文是"[MASK]"。']

        examples = [en_fil, en_gen, ch_fil, ch_gen, en_to_ch, ch_to_en]

        with gr.Blocks() as demo:
            gr.Markdown(
                """
                An Open Bilingual Pre-Trained Model. [Visit our github repo](https://github.com/THUDM/GLM-130B)
                GLM-130B uses two different mask tokens: `[MASK]` for short blank filling and `[gMASK]` for left-to-right long text generation. When the input does not contain any MASK token, `[gMASK]` will be automatically appended to the end of the text. We recommend that you use `[MASK]` to try text fill-in-the-blank to reduce wait time (ideally within seconds without queuing).
                """
            )

            with gr.Row():
                with gr.Column():
                    model_input = gr.Textbox(
                        lines=7, placeholder="Input something in English or Chinese", label="Input"
                    )
                    with gr.Row():
                        gen = gr.Button("Generate")
                        clr = gr.Button("Clear")

                outputs = gr.Textbox(lines=7, label="Output")

            gr.Markdown(
                """
                Generation Parameter
                """
            )
            with gr.Row():
                with gr.Column():
                    seed = gr.Slider(maximum=100000, value=1234, step=1, label="Seed")
                    out_seq_length = gr.Slider(
                        maximum=512, value=128, minimum=32, step=1, label="Output Sequence Length"
                    )
                with gr.Column():
                    min_gen_length = gr.Slider(maximum=64, value=0, step=1, label="Min Generate Length")
                    sampling_strategy = gr.Radio(
                        choices=["BeamSearchStrategy", "BaseStrategy"], value="BaseStrategy", label="Search Strategy"
                    )

            with gr.Row():
                with gr.Column():
                    # beam search
                    gr.Markdown(
                        """
                        BeamSearchStrategy
                        """
                    )
                    num_beams = gr.Slider(maximum=4, value=2, minimum=1, step=1, label="Number of Beams")
                    length_penalty = gr.Slider(maximum=1, value=1, minimum=0, label="Length Penalty")
                    no_repeat_ngram_size = gr.Slider(
                        maximum=5, value=3, minimum=1, step=1, label="No Repeat Ngram Size"
                    )
                with gr.Column():
                    # base search
                    gr.Markdown(
                        """
                        BaseStrategy
                        """
                    )
                    temperature = gr.Slider(maximum=1, value=1.0, minimum=0, label="Temperature")
                    topk = gr.Slider(maximum=40, value=0, minimum=0, step=1, label="Top K")
                    topp = gr.Slider(maximum=1, value=0.7, minimum=0, label="Top P")

            inputs = [
                model_input,
                seed,
                out_seq_length,
                min_gen_length,
                sampling_strategy,
                num_beams,
                length_penalty,
                no_repeat_ngram_size,
                temperature,
                topk,
                topp,
            ]
            gen.click(fn=predict, inputs=inputs, outputs=outputs)
            clr.click(fn=lambda value: gr.update(value=""), inputs=clr, outputs=model_input)

            gr_examples = gr.Examples(examples=examples, inputs=model_input)

        demo.launch(share=True)
    else:
        while True:
            info = [None, None, None, None, None, None, None, None, None, None, None]
            dist.broadcast_object_list(info, src=0)

            (
                text,
                seed,
                out_seq_length,
                min_gen_length,
                sampling_strategy,
                num_beams,
                length_penalty,
                no_repeat_ngram_size,
                temperature,
                topk,
                topp,
            ) = info

            predict(
                text,
                seed,
                out_seq_length,
                min_gen_length,
                sampling_strategy,
                num_beams,
                length_penalty,
                no_repeat_ngram_size,
                temperature,
                topk,
                topp,
            )


if __name__ == "__main__":
    args = initialize(extra_args_provider=add_generation_specific_args)

    with torch.no_grad():
        main(args)
