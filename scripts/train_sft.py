"""LoRA SFT on Tinker. Usage: train_sft.py CONFIG [--smoke]"""

from cotsl.cli import parse_args
from cotsl.config import SFTConfig, load_config
from cotsl.train.sft import train_sft


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, SFTConfig, smoke=args.smoke)
    print(train_sft(cfg, smoke=args.smoke))


if __name__ == "__main__":
    main()
