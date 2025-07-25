import argparse
import json
import logging
import os

import pandas as pd
from tqdm import tqdm

from game import WavelengthGame
from listeners import get_listener
from speakers import get_speaker
from utils import aggregate_metrics

logger = logging.getLogger(__name__)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--input_file", type=str, required=True, help="Path to the input file"
    )
    args.add_argument(
        "--output_file", type=str, required=True, help="Path to the output file"
    )
    args.add_argument(
        "--listener_type", type=str, default="l0_logprob", help="Type of listener model"
    )
    args.add_argument(
        "--listener_model", type=str, default=None, help="Path to the listener model"
    )
    args.add_argument(
        "--listener_port", type=int, required=True, help="Port for the listener model"
    )
    args.add_argument(
        "--speaker_type", type=str, default="l0", help="Type of speaker model"
    )
    args.add_argument(
        "--speaker_model", type=str, default=None, help="Path to the speaker model"
    )
    args.add_argument(
        "--speaker_port", type=int, required=True, help="Port for the speaker model"
    )
    args.add_argument(
        "--num_examples", type=int, default=None, help="Number of examples to evaluate"
    )
    args.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["listener", "speaker"],
        help="Mode to evaluate, play listener or speaker",
    )
    args.add_argument(
        "--num_alternatives",
        type=int,
        default=1,
        help="Number of alternatives to evaluate",
    )
    args.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for the speaker model",
    )
    args.add_argument(
        "--debug", action="store_true", help="Whether to use the debug mode"
    )
    args.add_argument(
        "--scale",
        type=int,
        default=5,
        help="Scale for the answer choices. Default is increments of 5.",
    )
    args.add_argument(
        "--use_null",
        type=str,
        default="true",
        help="Whether to use the null hypothesis",
    )
    args.add_argument(
        "--backend", type=str, default="vllm", help="Backend for the models"
    )
    args.add_argument(
        "--n_samples",
        type=int,
        default=32,
        help="Number of samples if use sampling to approximate the distribution",
    )
    args.add_argument(
        "--cache_filename", type=str, default=None, help="Path to the cache file"
    )
    return args.parse_args()


def parse_bool(s):
    assert s.lower() in ["true", "false"]
    return s.lower() == "true"


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    listener = None
    speaker = None
    if args.listener_model is not None:
        listener = get_listener(
            listener_type=args.listener_type,
            model=args.listener_model,
            port=args.listener_port,
            scale=args.scale,
            debug=args.debug,
            backend=args.backend,
            cache_filename=args.cache_filename,
            temperature=args.temperature,
        )
    if args.speaker_model is not None:
        speaker = get_speaker(
            speaker_type=args.speaker_type,
            model=args.speaker_model,
            port=args.speaker_port,
            scale=args.scale,
            num_alternatives=args.num_alternatives,
            temperature=args.temperature,
            debug=args.debug,
            backend=args.backend,
            cache_filename=args.cache_filename,
            n_samples=args.n_samples,
        )
    if args.speaker_model is None and args.listener_model is None:
        raise ValueError("Either speaker or listener model must be provided")
    if args.speaker_model is None and args.mode == "speaker":
        raise ValueError("Speaker model must be provided for speaker mode")
    if args.listener_model is None and args.mode == "listener":
        raise ValueError("Listener model must be provided for listener mode")

    data = pd.read_csv(args.input_file)[: args.num_examples]
    logger.info(f"Loaded {len(data)} examples from {args.input_file}")
    game = WavelengthGame(
        listener=listener,
        speaker=speaker,
        scale=args.scale,
        use_null=parse_bool(args.use_null),
        debug=args.debug,
    )

    outputs = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        example = row.to_dict()
        output = {
            "left": example["left"],
            "right": example["right"],
            "target": example["target"],
            "clue": example["clue"],
        }
        if args.mode == "listener":
            metrics = game.play_listener(
                left=output["left"],
                right=output["right"],
                clue=output["clue"],
                target=output["target"],
            )
        elif args.mode == "speaker":
            metrics = game.play_speaker(
                left=output["left"],
                right=output["right"],
                target=output["target"],
            )
        else:
            raise ValueError(f"Invalid mode: {args.mode}")
        output.update(metrics)
        outputs.append(output)

    logger.info(f"Finished evaluating {len(outputs)} examples")
    agg_metrics = aggregate_metrics(outputs)
    logger.info("======== Metrics =========")
    for metric_name, metric_value in agg_metrics.items():
        se_key = metric_name.replace("mean", "se")
        logger.info(f"{metric_name}: {metric_value:.2f} ± {agg_metrics[se_key]:.2f}")
    logger.info("======== End of Metrics =========")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = {
        "speaker": speaker.to_dict() if speaker is not None else None,
        "listener": listener.to_dict() if listener is not None else None,
        "use_null": parse_bool(args.use_null),
        "metrics": agg_metrics,
        "outputs": outputs,
    }

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved {len(outputs)} examples to {args.output_file}")


if __name__ == "__main__":
    main()
