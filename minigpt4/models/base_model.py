"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import re
import logging
import contextlib

from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
)

from minigpt4.common.dist_utils import download_cached_file
from minigpt4.common.utils import get_abs_path, is_url, cfg_from_yaml_file
from minigpt4.models.modeling_phi_local import PhiForCausalLM


class BaseModel(nn.Module):
    """Base class for models."""

    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return list(self.parameters())[-1].device

    def load_checkpoint(self, url_or_filename):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """

        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Build a pretrained model from default configuration file, specified by model_type.

        Args:
            - model_type (str): model type, specifying architecture and checkpoints.

        Returns:
            - model (nn.Module): pretrained or finetuned model, depending on the configuration.
        """
        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        model = cls.from_config(model_cfg)

        return model

    @classmethod
    def default_config_path(cls, model_type):
        assert (
                model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), "Unknown model type {}".format(model_type)
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])

    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_finetuned = cfg.get("load_finetuned", True)
        if load_finetuned:
            finetune_path = cfg.get("finetuned", None)
            assert (
                    finetune_path is not None
            ), "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
        else:
            # load pre-trained weights
            pretrain_path = cfg.get("pretrained", None)
            assert "Found load_finetuned is False, but pretrain_path is None."
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)

    def before_evaluation(self, **kwargs):
        pass

    def show_n_params(self, return_str=True):
        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
        else:
            return tot

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def force_trainable_fp32(self, tag=""):
        """AMP / GradScaler compatibility: make every *trainable* parameter fp32.

        peft >= 0.7 creates newly injected LoRA layers in the base-model dtype
        (fp16 for Phi-2 here), and torch.cuda.amp.GradScaler refuses to unscale
        fp16 parameter gradients -> "Attempting to unscale FP16 gradients".
        The original environment (peft 0.6 / RTX 3090) ended up with fp32
        adapters, so this restores that regime.  Frozen fp16 weights are left
        untouched, so the memory footprint of the frozen LLM is unchanged.
        """
        converted = []
        for name, param in self.named_parameters():
            if param.requires_grad and param.dtype == torch.float16:
                param.data = param.data.float()
                converted.append(name)
        logging.info(
            "[amp] cast %d trainable fp16 params to fp32%s%s",
            len(converted),
            (" " + tag) if tag else "",
            (" | examples: " + ", ".join(converted[:3])) if converted else "",
        )
        return converted

    @classmethod
    def init_pc_encoder(cls,  precision, freeze, pc_encoder_ckpt=None):
        logging.info('Loading pc encoder')

        from .pointbert.point_encoder import PointTransformer

        point_bert_config_addr = os.path.join(os.path.dirname(__file__),
                                              "pointbert/PointTransformer_8192point_2layer.yaml")
        point_bert_config = cfg_from_yaml_file(point_bert_config_addr)

        # use color
        point_bert_config.model.point_dims = 6
        use_max_pool = getattr(point_bert_config.model, "use_max_pool", False)  # * default is false

        point_encoder = PointTransformer(point_bert_config.model, use_max_pool=use_max_pool)

        if pc_encoder_ckpt is None:
            pc_encoder_ckpt = "./params_weight/pc_encoder/point_model_pcp_v2.pth"
            logging.warning(
                "[pc_encoder] pc_encoder_ckpt not specified in config, "
                "falling back to hardcoded default: %s. "
                "Set 'pc_encoder_ckpt' in your eval/train YAML to suppress this warning.",
                pc_encoder_ckpt
            )
        logging.info("[pc_encoder] Loading point encoder from: %s", pc_encoder_ckpt)
        point_encoder.load_checkpoint(pc_encoder_ckpt)


        if precision == "fp16":
            if freeze:
                # 冻结时转 FP16 可节省显存，不产生梯度
                convert_weights_to_fp16(point_encoder)
            else:
                # 训练时保持 FP32：GradScaler 要求梯度为 FP32
                logging.info('pc_encoder is trainable, keeping FP32 for GradScaler compatibility')

        logging.info(f"Using {point_encoder.point_dims} dim of points.")



        # if freeze:
        #     for name, param in point_encoder.named_parameters():
        #         param.requires_grad = False
        #     point_encoder = point_encoder.eval()
        #     point_encoder.train = disabled_train
        #
        #     logging.info("freeze pc encoder")
        #     print("Freeze pc encoder")

        logging.info('Loading pc encoder Done')
        return point_encoder

    def init_llm(cls, llama_model_path, low_resource=False, low_res_device=0, lora_r=0,
                 lora_target_modules=['query_key_value', 'dense'], **lora_kargs):
        logging.info('Loading LLAMA')
        llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=False)
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

        if low_resource:
            llama_model = PhiForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': low_res_device}
            )
        else:
            llama_model = PhiForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
            )

        if lora_r > 0:
            # llama_model = prepare_model_for_int8_training(llama_model)
            loraconfig = LoraConfig(
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_target_modules,
                **lora_kargs
            )
            llama_model = get_peft_model(llama_model, loraconfig)

            llama_model.print_trainable_parameters()
            for i, layer in enumerate(llama_model.model.model.layers):
                # layer.register_forward_hook(print_layer_output)
                # set trainable to True for the input_layernorm layer
                layer.self_attn.q_layernorm.weight.requires_grad = True
                layer.self_attn.k_layernorm.weight.requires_grad = True
                layer.post_layernorm.weight.requires_grad = True
                layer.input_layernorm.weight.requires_grad = True

                layer.self_attn.q_layernorm.weight.data = layer.self_attn.q_layernorm.weight.data.float()
                layer.self_attn.k_layernorm.weight.data = layer.self_attn.k_layernorm.weight.data.float()
                layer.post_layernorm.weight.data = layer.post_layernorm.weight.data.float()
                layer.input_layernorm.weight.data = layer.input_layernorm.weight.data.float()

                # 对偏置项进行类似操作
                if layer.self_attn.q_layernorm.bias is not None:
                    layer.self_attn.q_layernorm.bias.data = layer.self_attn.q_layernorm.bias.data.float()
                if layer.self_attn.k_layernorm.bias is not None:
                    layer.self_attn.k_layernorm.bias.data = layer.self_attn.k_layernorm.bias.data.float()
                if layer.input_layernorm.bias is not None:
                    layer.input_layernorm.bias.data = layer.input_layernorm.bias.data.float()

            llama_model.model.model.final_layernorm.weight.requires_grad = True
            llama_model.model.model.final_layernorm.weight.data = llama_model.model.model.final_layernorm.weight.data.float()
            if llama_model.model.model.final_layernorm.bias is not None:
                llama_model.model.model.final_layernorm.bias.data = llama_model.model.model.final_layernorm.bias.float()

        else:
            for name, param in llama_model.named_parameters():
                param.requires_grad = False

            # for i, layer in enumerate(llama_model.model.layers):
            #     # 如果层的索引小于5，则将该层的参数设置为可训练
            #     if i < 5:
            #         for param in layer.parameters():
            #             param.requires_grad = True
            #         # 将这些层的参数转换为FP32
            #         layer.to(torch.float32)
            for i, layer in enumerate(llama_model.model.layers):
                # layer.register_forward_hook(print_layer_output)
                # set trainable to True for the input_layernorm layer
                layer.self_attn.q_layernorm.weight.requires_grad = True
                layer.self_attn.k_layernorm.weight.requires_grad = True
                layer.post_layernorm.weight.requires_grad = True
                layer.input_layernorm.weight.requires_grad = True

                layer.self_attn.q_layernorm.weight.data = layer.self_attn.q_layernorm.weight.data.float()
                layer.self_attn.k_layernorm.weight.data = layer.self_attn.k_layernorm.weight.data.float()
                layer.post_layernorm.weight.data = layer.post_layernorm.weight.data.float()
                layer.input_layernorm.weight.data = layer.input_layernorm.weight.data.float()

                # 对偏置项进行类似操作
                if layer.self_attn.q_layernorm.bias is not None:
                    layer.self_attn.q_layernorm.bias.data = layer.self_attn.q_layernorm.bias.data.float()
                if layer.self_attn.k_layernorm.bias is not None:
                    layer.self_attn.k_layernorm.bias.data = layer.self_attn.k_layernorm.bias.data.float()
                if layer.input_layernorm.bias is not None:
                    layer.input_layernorm.bias.data = layer.input_layernorm.bias.data.float()

            llama_model.model.final_layernorm.weight.requires_grad = True
            llama_model.model.final_layernorm.weight.data = llama_model.model.final_layernorm.weight.data.float()
            if llama_model.model.final_layernorm.bias is not None:
                llama_model.model.final_layernorm.bias.data = llama_model.model.final_layernorm.bias.float()

        logging.info('Loading LLAMA Done')
        return llama_model, llama_tokenizer

    def load_from_pretrained(self, url_or_filename, key_remap=None):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        # ``key_remap`` adapts the checkpoint keys to the model's keys, e.g. to reach
        # LoRA-wrapped modules whose names peft >= 0.7 changed to ``*.base_layer.*``
        # (see ``remap_peft_base_layer_keys``). Without it such weights are silently
        # dropped by ``strict=False`` below.
        if key_remap is not None:
            state_dict = key_remap(state_dict, self.state_dict())

        msg = self.load_state_dict(state_dict, strict=False)

        # ``strict=False`` stays silent about unmatched keys; make the Q-Former
        # weights (which have no other source than this checkpoint) visible.
        warn_missing_qformer_keys(msg, url_or_filename)
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def remap_peft_base_layer_keys(state_dict, model_state_dict):
    """Remap checkpoint keys of LoRA-wrapped sub-modules onto peft's ``base_layer``.

    peft >= 0.7 keeps the layer a LoRA adapter is injected into as a ``base_layer``
    sub-module, so the parameter names of every LoRA target module become
    ``...<module>.base_layer.{weight,bias}``; peft <= 0.6 used an ``nn.Linear``
    subclass and kept the plain ``...<module>.{weight,bias}`` names.

    Checkpoints that provide pre-trained weights for the *unwrapped* naming (e.g. the
    BLIP-2 Q-Former, which is loaded into an already LoRA-injected Q-Former, or legacy
    full-model checkpoints) therefore no longer match and are silently dropped by
    ``load_state_dict(..., strict=False)`` - the affected modules would keep their
    random initialization. This mapping inserts ``.base_layer`` only when the target
    model actually exposes the wrapped key, so it is a no-op for un-wrapped modules
    (and for peft <= 0.6).

    Args:
        state_dict (dict): state dict of the checkpoint.
        model_state_dict (dict): ``model.state_dict()`` the checkpoint is loaded into.

    Returns:
        dict: state dict whose keys match the (possibly wrapped) model.
    """
    model_keys = set(model_state_dict.keys())
    remapped_state_dict = {}
    renamed = []
    for key, value in state_dict.items():
        if key in model_keys:
            remapped_state_dict[key] = value
            continue
        base_layer_key = re.sub(r"\.(weight|bias)$", r".base_layer.\1", key)
        if base_layer_key != key and base_layer_key in model_keys:
            remapped_state_dict[base_layer_key] = value
            renamed.append((key, base_layer_key))
        else:
            remapped_state_dict[key] = value
    if renamed:
        logging.info(
            "[load] remapped %d LoRA-wrapped key(s) onto peft base_layer, e.g. %s -> %s",
            len(renamed), renamed[0][0], renamed[0][1],
        )
    return remapped_state_dict


def warn_missing_qformer_keys(msg, source):
    """Warn about Q-Former weights silently skipped by ``load_state_dict(strict=False)``.

    The Q-Former (BLIP-2 pre-trained, frozen for stage >= 2 and hence never written
    into the stage checkpoints) only ever receives its *base* weights from
    ``blip2_pretrained_flant5xxl.pth``. Any key-naming mismatch would reset it to its
    random initialization without a visible error, so report it loudly instead.
    peft adapter parameters (``*.lora_*``) are excluded: they have no counterpart in a
    base checkpoint and are initialized by peft itself.

    Args:
        msg: return value of ``nn.Module.load_state_dict`` (uses ``missing_keys``).
        source (str): path/url of the checkpoint that was just loaded.

    Returns:
        list: the missing Q-Former keys (empty if everything matched).
    """
    missing_qformer = [
        k for k in msg.missing_keys
        if k.startswith("Qformer.") and ".lora_" not in k
    ]
    if missing_qformer:
        logging.warning(
            "[load] %d Q-Former key(s) MISSING while loading %s -> the Q-Former keeps "
            "its RANDOM initialization, e.g. %s",
            len(missing_qformer), source, missing_qformer[:3],
        )
    return missing_qformer


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()


    model.apply(_convert_weights_to_fp16)
