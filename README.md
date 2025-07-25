# WavelengthEval

This repository contains data and code for the paper ["On the Same Wavelength? Evaluating Pragmatic Reasoning in Language Models across Broad Concepts"](https://arxiv.org/abs/2509.06952) (Linlu Qiu, Cedegao E. Zhang, Joshua B. Tenenbaum, Yoon Kim, Roger P. Levy).


## Setup

```bash
pip install -r requirements.txt
```

## Run Experiments

We use the vLLM OpenAI-compatible server to host open-weights models. To serve a model, run:

```bash
vllm serve ${model_path} --port ${port}
```

We also support other backends, including Together, OpenAI, Gemini, and Anthropic. You'll need to set the required API keys as environment variables accordingly: `TOGETHER_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, and `ANTHROPIC_API_KEY`.

The `data/data.csv` contains the 50 concept pairs for our experiments, including 40 human responses for each concept pair.

To run evaluation, run:

```bash
python eval.py \
    --mode ${mode} \
    --input_file data/data.csv \
    --output_file results.json \
    --listener_type ${listener_type} \
    --listener_model ${listener_model_path} \
    --listener_port ${listener_port} \
    --speaker_type ${speaker_type} \
    --speaker_model ${speaker_model_path} \
    --speaker_port ${port} \
    --backend ${backend}
```

You need to specify the following key arguments:

- `mode`: Whether to evaluate the listener or speaker, choices: `listener` or `speaker`.
- `listener_type`: The type of listener, choices: `l0_logprob`, `l0_sample`, or `l0_sample_cot`.
- `listener_model`: The path to the listener model.
- `listener_port`: The port to the listener model.
- `speaker_type`: The type of speaker, choices: `l0` or `l0_cot`.
- `speaker_model`: The path to the speaker model.
- `speaker_port`: The port to the speaker model.
- `backend`: The backend to use, choices: `vllm`, `together`, `openai`, `gemini`, or `anthropic`.

You could change the following optional arguments:

- `num_examples`: The number of examples to evaluate, default is all examples.
- `num_alternatives`: The number of alternatives to generate for the speaker.
- `temperature`: The temperature to use for sampling. Default is 1.
- `scale`: The scale for the answer choices. Default is increments of 5.
- `use_null`: Whether to use the null hypothesis in the listener evaluation. Default is `true`.
- `n_samples`: The number of samples to use for the listener distribution. Default is 32.

When both a listener and a speaker are specified, we will evaluate the model with the base prompting method and its RSA version. If only one is specified, we will only evaluate the model without RSA.

