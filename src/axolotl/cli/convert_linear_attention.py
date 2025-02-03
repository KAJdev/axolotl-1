"""CLI to run training on a model."""

import logging
import os
from pathlib import Path
from typing import Union

import fire
from dotenv import load_dotenv
from transformers.hf_argparser import HfArgumentParser

from axolotl.cli.args import TrainerCliArgs
from axolotl.cli.art import print_axolotl_text_art
from axolotl.cli.checks import check_accelerate_default_config, check_user_token
from axolotl.cli.config import load_cfg
from axolotl.cli.utils import load_model_and_tokenizer
from axolotl.common.datasets import load_datasets
from axolotl.integrations.base import PluginManager
from axolotl.integrations.lolcats.linearize_attention import (
    remove_base_attention,
    toggle_attention,
)
from axolotl.utils.dict import DictDefault
from axolotl.utils.trainer import setup_trainer

LOG = logging.getLogger(__name__)


def do_linearize(cfg: DictDefault, cli_args: TrainerCliArgs) -> None:
    """
    Convert attention to linear attention and perform attention transfer via distillation.
    """
    print_axolotl_text_art()
    check_accelerate_default_config()
    check_user_token()

    # ensure quantization and peft are turned off (due to how we need to re-apply peft later)
    cfg.load_in_8bit = False
    cfg.load_in_4bit = False
    cfg.adapter = None

    # load model
    model, tokenizer = load_model_and_tokenizer(cfg=cfg)

    # convert attention
    from axolotl.integrations.lolcats.linearize_attention import convert_attention

    model = convert_attention(
        model, cfg.attention_config, train_attention=True, remove_base_attn=True
    )

    # Get datasets
    dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)
    train_dataset = dataset_meta.train_dataset
    eval_dataset = dataset_meta.eval_dataset
    total_num_steps = dataset_meta.total_num_steps

    # toggle attention to be trainable
    model = toggle_attention(model, train=True)

    # Setup trainer
    trainer = setup_trainer(
        cfg=cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=(model, None, None),
        tokenizer=tokenizer,
        processor=None,
        total_num_steps=total_num_steps,
    )

    # train
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    # drop base_attention + remove training attn
    model = toggle_attention(model, train=False)
    model = remove_base_attention(model)

    # NOTE: If in peft mode, consider whether to auto-merge

    # save model
    save_path = str(os.path.join(cfg.output_dir, "distilled"))
    tokenizer.save_pretrained(save_path)
    if hasattr(model, "config"):
        model.config.save_pretrained(save_path)

    safe_serialization = cfg.save_safetensors is True
    # NOTE: may need to consider other ways of saving due to multi-gpu etc
    model.save_pretrained(cfg.output_dir, safe_serialization=safe_serialization)

    # cleanup
    plugin_manager = PluginManager.get_instance()

    del model
    del tokenizer

    plugin_manager.post_train_unload(cfg)


def do_cli(config: Union[Path, str] = Path("examples/"), **kwargs) -> None:
    """
    Parses `axolotl` config, CLI args, and calls `do_train`.

    Args:
        config: Path to `axolotl` config YAML file.
        kwargs: Additional keyword arguments to override config file values.
    """
    # load cfg, force linearize and add plugin to linearize
    parsed_cfg = load_cfg(
        config,
        linearize=True,
        plugins=["axolotl.integrations.lolcats.LinearizePlugin"],
        **kwargs,
    )

    parser = HfArgumentParser(TrainerCliArgs)
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    do_linearize(parsed_cfg, parsed_cli_args)


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(do_cli)
