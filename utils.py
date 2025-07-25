import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def get_l1_dist(l0_listener_dict):
    # Pragmatic speaker: p(u | s)
    # (state, utterance) -> prob
    l1_speaker_dist = defaultdict(dict)
    for utterance, state_dist in l0_listener_dict.items():
        for state, prob in state_dist.items():
            l1_speaker_dist[state][utterance] = prob
    for state, utterance_dist in l1_speaker_dist.items():
        normalization = sum(utterance_dist.values())
        l1_speaker_dist[state] = {
            k: v / normalization for k, v in utterance_dist.items()
        }

    # Pragmatic listener p(s | u)
    # (utterance, state) -> prob
    l1_listener_dist = defaultdict(dict)
    for state, utterance_dist in l1_speaker_dist.items():
        for utterance, prob in utterance_dist.items():
            l1_listener_dist[utterance][state] = prob
    for utterance, state_dist in l1_listener_dist.items():
        normalization = sum(state_dist.values())
        l1_listener_dist[utterance] = {
            k: v / normalization for k, v in state_dist.items()
        }
    return l1_speaker_dist, l1_listener_dist


def print_distribution(distribution):
    max_value = max(distribution.values())
    scale_factor = 50
    for key, value in sorted(distribution.items(), key=lambda x: float(x[0])):
        bar_length = int((value / max_value) * scale_factor)
        logger.info(f"{key:>3} | {'█' * bar_length} {value:.2f}")
    logger.info("")


def aggregate_metrics(all_metrics, prefix=""):
    agg_metrics = defaultdict(list)
    for metrics in all_metrics:
        for metric_name, metric_value in metrics.items():
            agg_metrics[metric_name].append(metric_value)
    new_metrics = {}
    for metric_name, metric_values in agg_metrics.items():
        try:
            new_metrics[f"{prefix}{metric_name}_mean"] = np.mean(metric_values)
            new_metrics[f"{prefix}{metric_name}_se"] = np.std(metric_values) / np.sqrt(
                len(metric_values)
            )
        except Exception:
            continue
    return new_metrics
