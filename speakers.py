import logging
import re

from backends import get_backend

logger = logging.getLogger(__name__)


PREFIX = "<answer>"
SUFFIX = "</answer>"


INSTRUCTION = """# Context

You will be playing a language game. In each round, you will be presented a scale from 0 (left) to 100 (right). Each scale represents a spectrum between two opposite or contrasting concepts. One concept will be on the left side of the scale (towards 0), and the other concept will be on the right side (towards 100).

For each scale, you will see a target value, indicated by a number between 0 and 100. Your task is to think of a clue (often times a single word, or a very short phrase if necessary) that you believe best represents that specific target value's position on the scale, considering the two concepts. Essentially, you are trying to communicate where the target value lies on the spectrum using just your clue. A good clue should allow another person to approximately guess what the target value is.

Let's look at an example:

Imagine the scale is from Slow (0) to Fast (100).

If the target value is 85, you need a clue that represents something that is very fast, but perhaps not the absolute fastest. Good potential clues might include: 'professional sprinters', 'leopard', or 'Ferarri'. A clue like 'turtle' (which might be closer to 0) or 'bicycle' (which might be closer to 50) would be less appropriate for the target of 85 on this particular scale.

Here's another example:

Scale: Summer (0) to Winter (100).

Target value: 30. This target is closer to 'Summer' than to 'Winter'. Good clues might be: 'swimming', 'ice cream', or 'sunscreen'. Clues like 'wearing a shirt' (arguably closer to 50) or 'snow' (arguably closer to 100) would be less fitting for a target of 30 in this context.

Here're a few more examples:

Scale: Feels bad (0) to Feels good (100).
Target value: 70.
<answer>relaxing</answer>

Scale: Hard to spell (0) to Easy to spell (100).
Target value: 20.
<answer>Daenerys Targaryen</answer>

Scale: Red (0) to Yellow (100).
Target value: 50.
<answer>tangerine</answer>

You are encouraged to be concise when you come up with the clue. Try using a single word or a short phrase (a few words). You are not allowed to use more than 5 words for a clue. Your clue should convey a single thought. Do not combine multiple ideas. Do not use words that already appear in the two concepts or any synonyms (e.g., 'Piece' is not allowed for the 'Peaceful - Warlike' scale.) Do not use modifiers (words like 'but', 'very', 'almost', and 'slightly'). Do not use specific numbers or numeric values (including time, percentage, etc.). Do not use modifiers (words like 'but', 'very', 'almost', and 'slightly'). Do not use specific numbers or numeric values (including time, percentage, etc.).

Provide the best clue you can think of. The best clues maximize the chance that an average person can approximately guess the target value given the clue.

Format your response as:

<answer>your clue</answer>

# Problem

Scale: {left} (0) to {right} (100).
Target value: {target}."""


class Speaker:

    def __init__(
        self,
        model,
        port,
        backend="together",
        debug=False,
        temperature=1,
        max_tokens=2048,
        num_alternatives=1,
        cache_filename=None,
        **kwargs,
    ):
        self.backend = get_backend(
            backend,
            model=model,
            port=port,
            debug=debug,
            cache_filename=cache_filename,
        )
        self.num_alternatives = num_alternatives
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debug = debug

    def get_prompt(self, left, right, target):
        raise NotImplementedError

    def to_dict(self):
        return {
            "speaker_type": self.__class__.__name__,
            "model": self.backend.model,
            "prompt": self.get_prompt("left", "right", "target"),
            "num_alternatives": self.num_alternatives,
            "temperature": self.temperature,
        }


class L0Speaker(Speaker):

    def __init__(
        self,
        max_trial=10,
        scale=5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.stop = [SUFFIX]
        self.max_trial = max_trial
        self.answer_choices = [str(i) for i in range(0, 101, scale)]

    def get_prompt(self, left, right, target):
        return INSTRUCTION.format(
            left=left,
            right=right,
            target=target,
        )

    def parse_response(self, text):
        text = text.strip()
        text += SUFFIX
        try:
            response = re.findall(f"{PREFIX}(.*?){SUFFIX}", text, re.DOTALL)[0]
            response = response.replace("*", "")
            response = response.strip()
            return response
        except IndexError as e:
            logger.error(
                f"Error parsing response for {self.__class__.__name__}: {text}"
            )
            logger.error(f"Error: {e}")
            return None

    def get_alternatives_given_target(
        self,
        left,
        right,
        target,
        dedup=True,
    ):
        prompt = self.get_prompt(left, right, target)
        n = self.num_alternatives
        alternatives = []
        trial = 0
        while len(alternatives) < n:
            kwargs = {
                "temperature": self.temperature,
                "n": n,
                "stop": self.stop,
                "max_tokens": self.max_tokens,
            }
            if trial > 0:
                kwargs["request_id"] = trial
            choices = self.backend.generate(
                prompt,
                **kwargs,
            )
            for choice in choices:
                alternative = self.parse_response(choice)
                if alternative is None:
                    continue
                if dedup and alternative not in alternatives:
                    alternatives.append(alternative)
                elif not dedup:
                    alternatives.append(alternative)
            n = self.num_alternatives - len(alternatives)
            trial += 1
            if trial > self.max_trial:
                break
        return alternatives

    def get_alternatives(self, left, right, clue):
        alternatives = {}
        for value in self.answer_choices:
            alternatives[value] = self.get_alternatives_given_target(left, right, value)
        if self.debug:
            logger.info(
                f"======== Speaker alternatives: ({left}) {clue} ({right}) ========="
            )
            logger.info(alternatives)
        return alternatives


class L0COTSpeaker(L0Speaker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop = []

    def get_prompt(self, left, right, target):
        prompt = super().get_prompt(left, right, target)
        prompt += "\n\nLet's think step by step."
        return prompt


def get_speaker(speaker_type, **speaker_kwargs):
    if "cache_filename" in speaker_kwargs:
        speaker_kwargs["cache_filename"] = (
            f"speaker_{speaker_type}_{speaker_kwargs['cache_filename']}"
        )
    if speaker_type == "l0":
        return L0Speaker(**speaker_kwargs)
    elif speaker_type == "l0_cot":
        return L0COTSpeaker(**speaker_kwargs)
    else:
        raise ValueError(f"Invalid speaker type: {speaker_type}")
