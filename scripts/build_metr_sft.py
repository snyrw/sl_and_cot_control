"""Build METR-style ReasonIF SFT data (T_treat + optional T_ctrl(b)).
Usage: build_metr_sft.py CONFIG [--smoke]"""

import json

from cotsl.cli import parse_args, resolve_checkpoint
from cotsl.config import MetrSFTDataConfig, load_config
from cotsl.data.metr_sft import build_metr_sft


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, MetrSFTDataConfig, smoke=args.smoke)
    cfg = cfg.model_copy(update={"model": resolve_checkpoint(cfg.model)})
    stats = build_metr_sft(cfg, smoke=args.smoke)
    print(json.dumps({k: v for k, v in stats.items() if k != "generation"}, indent=1))


if __name__ == "__main__":
    main()
