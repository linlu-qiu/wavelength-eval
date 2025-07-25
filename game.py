import logging
from collections import Counter, defaultdict

import numpy as np

from utils import get_l1_dist

logger = logging.getLogger(__name__)


class WavelengthGame:

    def __init__(self, listener, speaker, scale=5, use_null=True, debug=False):
        self.listener = listener
        self.speaker = speaker
        self.scale = scale
        self.answer_choices = [str(i) for i in range(0, 101, scale)]
        self.int_answer_choices = np.array([int(i) for i in self.answer_choices])
        self.debug = debug
        self.use_null = use_null  # Whether to add the null hypothesis

        if self.debug:
            logger.info("======== Game settings =========")
            logger.info(f"Listener: {self.listener}")
            logger.info(f"Speaker: {self.speaker}")
            logger.info(f"Scale: {self.scale}")
            logger.info(f"Use null: {self.use_null}")
            logger.info("")

    def play_listener(self, left, right, clue, target):

        if self.debug:
            logger.info("======== Play listener =========")
            logger.info(f"Left: {left}")
            logger.info(f"Right: {right}")
            logger.info(f"Clue: {clue}")
            logger.info(f"Target value: {target}")
            logger.info("")

        l0_distribution = self.listener.get_distribution(left, right, clue)
        all_metrics = {}
        l0_metrics = self.compute_listener_metrics(
            l0_distribution, target, prefix="l0_"
        )
        all_metrics.update(l0_metrics)
        if self.speaker is None:
            return all_metrics

        alternatives_dict = self.speaker.get_alternatives(left, right, clue)

        all_words = set()
        for _, values in alternatives_dict.items():
            all_words |= set(values)
        all_words = list(all_words)

        l0_listener_dict = defaultdict(dict)
        l0_listener_dict[clue] = l0_distribution
        for word in all_words:
            distribution = self.listener.get_distribution(left, right, word)
            l0_listener_dict[word] = distribution

        if self.use_null:
            l0_listener_dict[None] = {
                answer: 1 / len(self.answer_choices) for answer in self.answer_choices
            }

        _, l1_listener_dist = get_l1_dist(l0_listener_dict)

        l1_distribution = l1_listener_dist[clue]
        l1_metrics = self.compute_listener_metrics(
            l1_distribution, target, prefix="l1_"
        )
        all_metrics.update(l1_metrics)
        all_metrics["listener_distributions"] = l0_listener_dict
        all_metrics["alternatives"] = alternatives_dict
        return all_metrics

    def compute_listener_metrics(self, distribution, target, prefix=""):
        answer = max(distribution, key=distribution.get)
        diff = abs(float(answer) - float(target))
        expected_answer = sum(
            distribution[answer] * float(answer) for answer in distribution
        )
        expected_diff = abs(expected_answer - float(target))
        metrics = {
            f"{prefix}answer": answer,
            f"{prefix}diff": diff,
            f"{prefix}expected_diff": expected_diff,
            f"{prefix}expected_answer": expected_answer,
            f"{prefix}distribution": distribution,
        }
        if self.debug:
            logger.info(f"======== {prefix} Metrics =========")
            for key, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"{key}: {value:.2f}")
        return metrics

    def play_speaker(self, left, right, target):

        if self.debug:
            logger.info("======== Play speaker =========")
            logger.info(f"Left: {left}")
            logger.info(f"Right: {right}")
            logger.info(f"Target: {target}")
            logger.info("")

        alternatives = self.speaker.get_alternatives_given_target(
            left=left, right=right, target=target, dedup=False
        )
        prior = Counter(alternatives)
        prior = {k: v / len(alternatives) for k, v in prior.items()}
        l0_clue = np.random.choice(alternatives)
        l0_metrics = self.compute_speaker_metrics(prior, l0_clue, prefix="l0_")
        all_metrics = {}
        all_metrics.update(l0_metrics)

        if self.listener is None:
            return all_metrics

        l0_listener_dict = defaultdict(dict)
        for clue in set(alternatives):
            distribution = self.listener.get_distribution(left, right, clue)
            distribution = {k: v * prior[clue] for k, v in distribution.items()}
            l0_listener_dict[clue] = distribution

        l1_speaker_dist, _ = get_l1_dist(l0_listener_dict)
        l1_dist = l1_speaker_dist[str(target)]
        if not l1_dist:
            logger.info(f"l1_dist is empty for target {target}")
            l1_clue = np.random.choice(list(prior.keys()), p=list(prior.values()))
        else:
            l1_clue = np.random.choice(list(l1_dist.keys()), p=list(l1_dist.values()))

        l1_metrics = self.compute_speaker_metrics(l1_dist, l1_clue, prefix="l1_")
        all_metrics.update(l1_metrics)
        all_metrics["listener_distributions"] = l0_listener_dict
        all_metrics["alternatives"] = alternatives
        return all_metrics

    def compute_speaker_metrics(self, distribution, clue, prefix=""):
        return {
            f"{prefix}clue": clue,
            f"{prefix}distribution": distribution,
        }
