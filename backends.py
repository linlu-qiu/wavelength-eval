import logging
import os
import time
from copy import deepcopy

from openai import OpenAI
from together import Together
from transformers import AutoTokenizer

from cache_utils import SimpleCache

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


logger = logging.getLogger(__name__)


def serialize_dict(item):
    item_dict = {}
    dtypes = [str, int, float, bool]
    if any(isinstance(item, dtype) for dtype in dtypes):
        return item
    if item is None:
        return item
    if isinstance(item, list):
        return [serialize_dict(item) for item in item]
    for key, value in dict(item).items():
        item_dict[key] = serialize_dict(value)
    return item_dict


class Backend:

    def __init__(self, model, debug=False, cache_dir="cache", cache_filename=None):
        self.model = model
        self.debug = debug
        os.makedirs(os.path.join(CACHE_DIR, cache_dir), exist_ok=True)
        if cache_filename is None:
            model_name = self.model.replace("/", "_").strip("_")
            cache_filename = f"{model_name}.jsonl"
        self.cache_file = os.path.join(CACHE_DIR, cache_dir, cache_filename)
        self.cache = SimpleCache(self.cache_file)

    def generate(self, prompt, **kwargs):
        raise NotImplementedError


class VLLMModel(Backend):

    def __init__(
        self,
        model,
        port,
        debug=False,
        cache_filename=None,
    ):
        super().__init__(model, debug, cache_dir="vllm", cache_filename=cache_filename)
        self.client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="token-abc123",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def maybe_apply_chat_template(self, prompt, target=None):
        if self.tokenizer.chat_template:
            if target is None:
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            else:
                prompt = self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": target},
                    ],
                    tokenize=False,
                    enable_thinking=False,
                )
        else:
            assert target is None, "Chat template is not set for the tokenizer"
        if self.tokenizer.bos_token and prompt.startswith(self.tokenizer.bos_token):
            prompt = prompt[len(self.tokenizer.bos_token) :]
        return prompt

    def generate(self, prompt, **kwargs):
        prompt = self.maybe_apply_chat_template(prompt)
        request = {
            "model": self.model,
            "prompt": prompt,
            **kwargs,
        }
        raw_request = deepcopy(request)
        if "request_id" in request:
            request.pop("request_id")
        response = self.cache.get(raw_request)
        if response is None:
            response = self.client.completions.create(**request)
            response = serialize_dict(response)
            self.cache.set(raw_request, response)
        return [choice["text"] for choice in response["choices"]]

    def get_logprobs(self, prompt, targets, logprobs_type="cumulative"):

        def get_cumulative_logprob(token_ids, logprobs):
            cumulative_logprob = 0
            for token_id, logprob in zip(token_ids, logprobs):
                cumulative_logprob += logprob[str(token_id)]["logprob"]
            return cumulative_logprob

        def get_logprob(prompt, target):
            combined_prompt = self.maybe_apply_chat_template(prompt, target)
            original_prompt = self.maybe_apply_chat_template(prompt)
            tokens = self.tokenizer.encode(combined_prompt)
            request = {
                "model": self.model,
                "prompt": combined_prompt,
                "temperature": 1,
                "max_tokens": 1,
                "extra_body": {
                    "prompt_logprobs": 1,
                },
            }
            response = self.cache.get(request)
            if response is None:
                response = self.client.completions.create(**request)
                response = serialize_dict(response)
                self.cache.set(request, response)

            tokens = self.tokenizer.encode(combined_prompt)[1:]
            prompt_tokens = self.tokenizer.encode(original_prompt)[1:]
            assert prompt_tokens == tokens[: len(prompt_tokens)]
            completion_tokens = tokens[len(prompt_tokens) :]
            token_logprobs = response["choices"][0]["prompt_logprobs"][1:]
            completion_logprobs = token_logprobs[len(prompt_tokens) :]

            if logprobs_type == "cumulative":
                return get_cumulative_logprob(completion_tokens, completion_logprobs)
            if logprobs_type == "completion_cumulative":
                return get_cumulative_logprob(tokens, token_logprobs)
            if logprobs_type == "avg":
                cumulative_logprob = get_cumulative_logprob(tokens, token_logprobs)
                return cumulative_logprob / len(tokens)
            if logprobs_type == "completion_avg":
                cumulative_logprob = get_cumulative_logprob(
                    completion_tokens, completion_logprobs
                )
                return cumulative_logprob / len(completion_tokens)
            else:
                raise ValueError(f"Invalid logprobs type: {logprobs_type}")

        logprobs = []
        if self.debug:
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Targets: {targets}")
        for choice in targets:
            logprob = get_logprob(prompt, choice)
            logprobs.append(logprob)
        return logprobs


class TogetherModel(Backend):

    def __init__(
        self,
        model,
        debug=False,
        cache_filename=None,
        max_trial=5,
        **kwargs,
    ):
        super().__init__(
            model, debug, cache_dir="together", cache_filename=cache_filename
        )
        self.client = Together()
        self.max_trial = max_trial

    def generate(self, prompt, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        request = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }
        raw_request = deepcopy(request)
        if "request_id" in request:
            request.pop("request_id")
        response = self.cache.get(raw_request)
        if response is None:
            trial = 0
            while trial < self.max_trial:
                try:
                    response = self.client.chat.completions.create(**request)
                    response = serialize_dict(response)
                    self.cache.set(raw_request, response)
                    break
                except Exception as e:
                    logger.error(f"Error calling {self.model}: {e}")
                    trial += 1
                    time.sleep(30)
            if trial == self.max_trial:
                raise ValueError("Failed to get response from Together")
        return [choice["message"]["content"] for choice in response["choices"]]

    def get_logprobs(self, prompt, targets, logprobs_type="cumulative"):

        def get_logprob(prompt, target):

            assert logprobs_type == "cumulative"

            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ]
            request = {
                "model": self.model,
                "messages": messages,
                "logprobs": 1,
                "max_tokens": 1,
                "echo": True,
            }
            response = self.cache.get(request)
            if response is None:
                response = self.client.chat.completions.create(**request)
                response = serialize_dict(response)
                self.cache.set(request, response)
            token_logprobs = response["prompt"][0]["logprobs"]["token_logprobs"][1:]
            return sum(token_logprobs)

        logprobs = []
        if self.debug:
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Targets: {targets}")
        for choice in targets:
            logprob = get_logprob(prompt, choice)
            logprobs.append(logprob)
        return logprobs


class OpenAIModel(Backend):

    def __init__(
        self,
        model,
        debug=False,
        cache_filename=None,
        max_trial=3,
        **kwargs,
    ):
        super().__init__(
            model, debug, cache_dir="openai", cache_filename=cache_filename
        )
        self.client = OpenAI()
        self.max_n = 8
        self.max_trial = max_trial

    def generate(self, prompt, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        request = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }
        raw_request = deepcopy(request)
        if "request_id" in request:
            request.pop("request_id")
        response = self.cache.get(raw_request)
        if response is None:
            trial = 0
            while trial < self.max_trial:
                try:
                    requests = self.split_request(request)
                    responses = []
                    for request in requests:
                        response = self.client.chat.completions.create(**request)
                        responses.append(serialize_dict(response))
                    response = self.merge_responses(responses)
                    self.cache.set(raw_request, response)
                    break
                except Exception as e:
                    logger.error(f"Error calling {self.model}: {e}")
                    trial += 1
                    time.sleep(30)
            if trial == self.max_trial and response is None:
                raise Exception(
                    f"Failed to get response from {self.model} after {self.max_trial} retries"
                )
        return [choice["message"]["content"] for choice in response["choices"]]

    def split_request(self, request):
        n = request.get("n", 1)
        if n > self.max_n:
            real_requests = []
            for _ in range(0, n, self.max_n):
                copy_request = deepcopy(request)
                copy_request["n"] = self.max_n
                real_requests.append(copy_request)
            return real_requests
        else:
            return [request]

    def merge_responses(self, responses):
        choices = []
        for response in responses:
            choices.extend(response["choices"])
        responses[0]["choices"] = choices
        return responses[0]


class GeminiModel(OpenAIModel):

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.client = OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def generate(self, prompt, **kwargs):
        if "stop" in kwargs and not kwargs["stop"]:
            kwargs.pop("stop")
        return super().generate(prompt, **kwargs)


class AnthropicModel(OpenAIModel):

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.client = OpenAI(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            base_url="https://api.anthropic.com/v1/",
        )
        self.max_n = 1

    def generate(self, prompt, **kwargs):
        kwargs["extra_body"] = {
            "thinking": {"type": "disabled"},
        }
        if "stop" in kwargs:
            kwargs["stop"] = [s for s in kwargs["stop"] if s.strip() != ""]
        return super().generate(prompt, **kwargs)


def get_backend(backend, **kwargs):
    if backend == "vllm":
        return VLLMModel(**kwargs)
    elif backend == "together":
        return TogetherModel(**kwargs)
    elif backend == "openai":
        return OpenAIModel(**kwargs)
    elif backend == "gemini":
        return GeminiModel(**kwargs)
    elif backend == "anthropic":
        return AnthropicModel(**kwargs)
    else:
        raise ValueError(f"Invalid backend: {backend}")
