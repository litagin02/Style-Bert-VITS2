import glob
import os
import re
from pathlib import Path
from typing import Any, Optional, Union

import torch

from style_bert_vits2.logging import logger


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    skip_optimizer: bool = False,
    for_infer: bool = False,
) -> tuple[torch.nn.Module, Optional[torch.optim.Optimizer], float, int]:
    """
    指定されたパスからチェックポイントを読み込み、モデルとオプティマイザーを更新する。

    Args:
        checkpoint_path (Union[str, Path]): チェックポイントファイルのパス
        model (torch.nn.Module): 更新するモデル
        optimizer (Optional[torch.optim.Optimizer]): 更新するオプティマイザー。None の場合は更新しない
        skip_optimizer (bool): オプティマイザーの更新をスキップするかどうかのフラグ
        for_infer (bool): 推論用に読み込むかどうかのフラグ

    Returns:
        tuple[torch.nn.Module, Optional[torch.optim.Optimizer], float, int]: 更新されたモデルとオプティマイザー、学習率、イテレーション回数
    """

    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    logger.info(
        f"Loading model and optimizer at iteration {iteration} from {checkpoint_path}"
    )
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict.get("optimizer") is not None
    ):
        try:
            # Check if parameter groups match
            saved_groups = len(checkpoint_dict["optimizer"]["param_groups"])
            current_groups = len(optimizer.param_groups)
            
            if saved_groups == current_groups:
                # Try to load optimizer state
                optimizer.load_state_dict(checkpoint_dict["optimizer"])
                logger.info("Successfully loaded optimizer state")
            else:
                logger.warning(
                    f"Parameter group mismatch: saved {saved_groups} vs current {current_groups}. "
                    "Skipping optimizer state loading. This may be due to model freezing changes."
                )
        except ValueError as e:
            logger.warning(
                f"Failed to load optimizer state: {e}\n"
                "This may be due to changes in model architecture or freezing settings. "
                "Training will continue with fresh optimizer state."
            )
        except Exception as e:
            logger.error(f"Unexpected error loading optimizer state: {e}")

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # assert "emb_g" not in k
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (
                saved_state_dict[k].shape,
                v.shape,
            )
        except:
            # For upgrading from the old version
            if "ja_bert_proj" in k:
                v = torch.zeros_like(v)
                logger.warning(
                    f"Seems you are using the old version of the model, the {k} is automatically set to zero for backward compatibility"
                )
            elif "enc_q" in k and for_infer:
                continue
            else:
                logger.error(f"{k} is not in the checkpoint {checkpoint_path}")

            new_state_dict[k] = v

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)

    logger.info(f"Loaded '{checkpoint_path}' (iteration {iteration})")

    return model, optimizer, learning_rate, iteration


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Union[torch.optim.Optimizer, torch.optim.AdamW],
    learning_rate: float,
    iteration: int,
    checkpoint_path: Union[str, Path],
) -> None:
    """
    モデルとオプティマイザーの状態を指定されたパスに保存する。

    Args:
        model (torch.nn.Module): 保存するモデル
        optimizer (Union[torch.optim.Optimizer, torch.optim.AdamW]): 保存するオプティマイザー
        learning_rate (float): 学習率
        iteration (int): イテレーション回数
        checkpoint_path (Union[str, Path]): 保存先のパス
    """
    logger.info(
        f"Saving model and optimizer state at iteration {iteration} to {checkpoint_path}"
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def clean_checkpoints(
    model_dir_path: Union[str, Path] = "logs/44k/",
    n_ckpts_to_keep: int = 2,
    sort_by_time: bool = True,
) -> None:
    """
    指定されたディレクトリから古いチェックポイントを削除して空き容量を確保する

    Args:
        model_dir_path (Union[str, Path]): モデルが保存されているディレクトリのパス
        n_ckpts_to_keep (int): 保持するチェックポイントの数（G_0.pth と D_0.pth を除く）
        sort_by_time (bool): True の場合、時間順に削除。False の場合、名前順に削除
    """

    ckpts_files = [
        f
        for f in os.listdir(model_dir_path)
        if os.path.isfile(os.path.join(model_dir_path, f))
    ]

    def name_key(_f: str) -> int:
        return int(re.compile("._(\\d+)\\.pth").match(_f).group(1))  # type: ignore

    def time_key(_f: str) -> float:
        return os.path.getmtime(os.path.join(model_dir_path, _f))

    sort_key = time_key if sort_by_time else name_key

    def x_sorted(_x: str) -> list[str]:
        return sorted(
            [f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pth")],
            key=sort_key,
        )

    to_del = [
        os.path.join(model_dir_path, fn)
        for fn in (
            x_sorted("G_")[:-n_ckpts_to_keep]
            + x_sorted("D_")[:-n_ckpts_to_keep]
            + x_sorted("WD_")[:-n_ckpts_to_keep]
            + x_sorted("DUR_")[:-n_ckpts_to_keep]
        )
    ]

    def del_info(fn: str) -> None:
        return logger.info(f"Free up space by deleting ckpt {fn}")

    def del_routine(x: str) -> list[Any]:
        return [os.remove(x), del_info(x)]

    [del_routine(fn) for fn in to_del]


def get_latest_checkpoint_path(
    model_dir_path: Union[str, Path], regex: str = "G_*.pth"
) -> str:
    """
    指定されたディレクトリから最新のチェックポイントのパスを取得する

    Args:
        model_dir_path (Union[str, Path]): モデルが保存されているディレクトリのパス
        regex (str): チェックポイントのファイル名の正規表現

    Returns:
        str: 最新のチェックポイントのパス
    """

    f_list = glob.glob(os.path.join(str(model_dir_path), regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    try:
        x = f_list[-1]
    except IndexError:
        raise ValueError(f"No checkpoint found in {model_dir_path} with regex {regex}")

    return x
