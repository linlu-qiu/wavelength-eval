import logging
import re
from collections import Counter

from scipy.special import softmax

from backends import get_backend
from utils import print_distribution

logger = logging.getLogger(__name__)


PREFIX = "<answer>"
SUFFIX = "</answer>"


INSTRUCTION = """# Context

You will be playing a language game. In each round, you will be presented a scale from 0 (left) to 100 (right). Each scale represents a spectrum between two opposite or contrasting concepts. One concept will be on the left side of the scale (towards 0), and the other concept will be on the right side (towards 100).

For each scale, you will see a clue—a single word or a short phrase—that someone else has provided. Your task is to guess a value between 0 and 100 (in increments of {scale}) that you believe best represents where that clue falls on the scale between the two concepts. Essentially, you are trying to interpret where on the spectrum the clue giver intended their clue to represent. A good guess should be close to the target value that the clue giver was trying to communicate. (The clue giver chose the clue with respect to the value provided to them.)

Let's look at an example:

Imagine the scale is from Slow (0) to Fast (100).

If the clue is 'race car', you would likely guess a high value, perhaps around 85, since a race car represents something that is very fast, but not the absolute fastest thing possible. If the clue had been 'turtle' instead, you might guess a value closer to 10, or if it had been 'bicycle', you might guess a value closer to 50 on this particular scale.

Here's another example:

Scale: Summer (0) to Winter (100). 
Clue: 'ice cream'.

This clue is more strongly associated with 'Summer' than with 'Winter'. A reasonable guess might be around 30. While people certainly eat ice cream year-round, it is especially popular during hot summer days. If the clue had been 'wearing a shirt' you might guess a value closer to 50, or if it had been 'snow' you might guess a value closer to 90 or 100 in this context.

Here're a few more examples:

Scale: Feels bad (0) to Feels good (100).
Clue: relaxing.
<answer>70</answer>

Scale: Hard to spell (0) to Easy to spell (100).
Clue: Daenerys Targaryen.
<answer>20</answer>

Scale: Red (0) to Yellow (100).
Clue: tangerine.
<answer>50</answer>

Provide your best estimate (in increments of {scale}, so the possible values are {possible_values}) carefully. The goal is to get as close as possible to the target value that the clue giver intended to communicate. If you find a clue confusing, meaningless, or hard to understand, please still make an educated guess. Some clues may not be very good, but please always make an honest attempt.

Format your response as:

<answer>your guess</answer>

# Problem

Scale: {left} (0) to {right} (100).
Clue: {clue}."""


SAMPLE_INSTRUCTION = """# Context

You will be playing a language game. In each round, you will be presented a scale from 0 (left) to 100 (right). Each scale represents a spectrum between two opposite or contrasting concepts. One concept will be on the left side of the scale (towards 0), and the other concept will be on the right side (towards 100).

For each scale, you will see a clue—a single word or a short phrase—that someone else has provided. Your task is to guess a value between 0 and 100 (any integer) that you believe best represents where that clue falls on the scale between the two concepts. Essentially, you are trying to interpret where on the spectrum the clue giver intended their clue to represent. A good guess should be close to the target value that the clue giver was trying to communicate. (The clue giver chose the clue with respect to the value provided to them.)

Let's look at an example:

Imagine the scale is from Slow (0) to Fast (100).

If the clue is 'race car', you would likely guess a high value, perhaps around 85, since a race car represents something that is very fast, but not the absolute fastest thing possible. If the clue had been 'turtle' instead, you might guess a value closer to 10, or if it had been 'bicycle', you might guess a value closer to 50 on this particular scale.

Here's another example:

Scale: Summer (0) to Winter (100). 
Clue: 'ice cream'.

This clue is more strongly associated with 'Summer' than with 'Winter'. A reasonable guess might be around 30. While people certainly eat ice cream year-round, it is especially popular during hot summer days. If the clue had been 'wearing a shirt' you might guess a value closer to 50, or if it had been 'snow' you might guess a value closer to 90 or 100 in this context.

Here're a few more examples:

Scale: Feels bad (0) to Feels good (100).
Clue: relaxing.
<answer>70</answer>

Scale: Hard to spell (0) to Easy to spell (100).
Clue: Daenerys Targaryen.
<answer>20</answer>

Scale: Red (0) to Yellow (100).
Clue: tangerine.
<answer>50</answer>

Provide your best estimate (an integer between 0 and 100) carefully. The goal is to get as close as possible to the target value that the clue giver intended to communicate. If you find a clue confusing, meaningless, or hard to understand, please still make an educated guess. Some clues may not be very good, but please always make an honest attempt.

Format your response as:

<answer>your guess</answer>

# Problem

Scale: {left} (0) to {right} (100).
Clue: {clue}."""


class Listener:

    def __init__(
        self,
        model,
        port,
        backend="together",
        debug=False,
        cache_filename=None,
        temperature=1,
        **kwargs,
    ):
        self.backend = get_backend(
            backend,
            model=model,
            port=port,
            debug=debug,
            cache_filename=cache_filename,
        )
        self.debug = debug
        self.temperature = temperature

    def get_prompt(self, left, right, clue):
        raise NotImplementedError

    def get_distribution(self, left, right, clue):
        raise NotImplementedError

    def to_dict(self):
        return {
            "listener_type": self.__class__.__name__,
            "model": self.backend.model,
            "prompt": self.get_prompt("left", "right", "clue"),
        }


class L0LogprobListener(Listener):

    def __init__(
        self,
        scale=5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.answer_choices = [str(i) for i in range(0, 101, scale)]

    def get_prompt(self, left, right, clue):
        possible_values = ", ".join(self.answer_choices)
        prompt = INSTRUCTION.format(
            left=left,
            right=right,
            clue=clue,
            scale=self.scale,
            possible_values=possible_values,
        )
        return prompt

    def get_distribution(self, left, right, clue):
        prompt = self.get_prompt(left, right, clue)
        answer_choice_strs = [f"{PREFIX}{ans}{SUFFIX}" for ans in self.answer_choices]
        logprobs = self.backend.get_logprobs(prompt, answer_choice_strs)
        probs = softmax(logprobs)
        distribution = {ans: prob for ans, prob in zip(self.answer_choices, probs)}
        if self.debug:
            logger.info(
                f"========= Distribution for ({left}) {clue} ({right}) ========="
            )
            print_distribution(distribution)
        return distribution


class L0SampleListener(Listener):

    def __init__(
        self, n_samples=32, max_trial_per_sample=10, max_tokens=2048, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples
        self.max_trial_per_sample = max_trial_per_sample
        self.model_suffix = SUFFIX
        self.stop = [SUFFIX]
        self.max_tokens = max_tokens

    def get_prompt(self, left, right, clue):
        prompt = SAMPLE_INSTRUCTION.format(left=left, right=right, clue=clue)
        return prompt

    def parse_response(self, text):
        text = text.strip()
        text += self.model_suffix
        try:
            response = re.findall(f"{PREFIX}(.*?){SUFFIX}", text, re.DOTALL)[0]
            response = response.replace("*", "")
            response = response.strip()
            float(response)
            return response
        except Exception as e:
            logger.error(
                f"Error parsing response for {self.__class__.__name__} due to {e}: {text}"
            )
            return None

    def get_distribution(self, left, right, clue):
        samples = []
        trial = 0
        while len(samples) < self.n_samples:
            prompt = self.get_prompt(left, right, clue)
            kwargs = {
                "temperature": self.temperature,
                "n": 1,
                "stop": self.stop,
                "max_tokens": self.max_tokens,
            }
            if trial > 0:
                kwargs["request_id"] = trial
            choices = self.backend.generate(prompt, **kwargs)
            sample = self.parse_response(choices[0])
            if sample:
                samples.append(sample)
            if trial > self.max_trial_per_sample * self.n_samples:
                break
            trial += 1

        counts = Counter(samples)
        if len(samples) != self.n_samples:
            logger.warning(
                f"Warning: Only {len(counts)} samples out of {self.n_samples} were generated"
            )
        distribution = {k: v / len(samples) for k, v in counts.items()}
        if self.debug:
            logger.info(
                f"========= Distribution for ({left}) {clue} ({right}) ========="
            )
            print_distribution(distribution)
        return distribution

    def to_dict(self):
        data_dict = super().to_dict()
        data_dict["n_samples"] = self.n_samples
        data_dict["max_trial_per_sample"] = self.max_trial_per_sample
        return data_dict


class L0SampleCOTListener(L0SampleListener):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop = []
        self.model_suffix = ""

    def get_prompt(self, left, right, clue):
        prompt = super().get_prompt(left, right, clue)
        prompt += "\n\nLet's think step by step."
        return prompt


def get_listener(listener_type, **listener_kwargs):
    if "cache_filename" in listener_kwargs:
        listener_kwargs["cache_filename"] = (
            f"listener_{listener_type}_{listener_kwargs['cache_filename']}"
        )
    if listener_type == "l0_logprob":
        return L0LogprobListener(**listener_kwargs)
    elif listener_type == "l0_sample":
        return L0SampleListener(**listener_kwargs)
    elif listener_type == "l0_sample_cot":
        return L0SampleCOTListener(**listener_kwargs)
    else:
        raise ValueError(f"Invalid listener type: {listener_type}")
