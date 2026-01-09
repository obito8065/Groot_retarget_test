# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import Optional
import time
import torch
import transformers
from torch.utils.data import Dataset, Sampler
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
    set_rng_state_for_device,
    logging,
    safe_globals,
    random,
    np,
    is_torch_xla_available,
    is_torch_npu_available,
    ParallelMode,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_musa_available,
)

logger = logging.get_logger(__name__)
original_load_rng_state = transformers.Trainer._load_rng_state

def patched_load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        with safe_globals():
            checkpoint_rng_state = torch.load(rng_file, weights_only=False)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if is_torch_xla_available():
            xm.set_rng_state(checkpoint_rng_state["xla"])

        is_distributed = self.args.parallel_mode == ParallelMode.DISTRIBUTED
        if torch.cuda.is_available():
            set_rng_state_for_device("CUDA", torch.cuda, checkpoint_rng_state, is_distributed)
        if is_torch_npu_available():
            set_rng_state_for_device("NPU", torch.npu, checkpoint_rng_state, is_distributed)
        if is_torch_hpu_available():
            set_rng_state_for_device("HPU", torch.hpu, checkpoint_rng_state, is_distributed)
        if is_torch_mlu_available():
            set_rng_state_for_device("MLU", torch.mlu, checkpoint_rng_state, is_distributed)
        if is_torch_musa_available():
            set_rng_state_for_device("MUSA", torch.musa, checkpoint_rng_state, is_distributed)

transformers.Trainer._load_rng_state = patched_load_rng_state

class BaseSampler(Sampler):
    """Sampler for dataset, which enables `set_epoch` for Dataset.
    `set_epoch` will be called by huggingface Trainer at the end of each epoch.
    `shuffle` is also supported for training set shuffling
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # must not add rank here, or randomization will be different for each rank
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            # this is important for dataset
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


class DualBrainTrainer(transformers.Trainer):
    def __init__(self, **kwargs):
        self.compute_dtype = kwargs.pop("compute_dtype")
        super().__init__(**kwargs)

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(self.eval_dataset, shuffle=False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs)
        loss = outputs["loss"]

        with torch.no_grad():
            start_time = time.time()
            logs: dict[str, float] = {}
            if self.state.global_step % self.state.logging_steps == 0:
                should_log = True
            else:
                should_log = False
            if should_log and 'latent_loss' in outputs:
                logs['latent_loss'] = round(outputs['latent_loss'].item(), 4)
                logs['flow_loss'] = round(outputs['flow_loss'].item(), 4)
                for callback in self.callback_handler.callbacks:
                    if 'TensorBoard' in str(callback):
                        getattr(callback, "on_log")(
                            self.args,
                            self.state,
                            self.control,
                            model=self.model,
                            logs=logs
                        )

        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        重写 prediction_step，确保评估时也使用 model(inputs) 的方式调用模型
        """
        # model.eval()
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        # 返回 (loss, logits, labels) 格式
        # 对于你的模型，可能不需要 logits 和 labels
        return (loss, None, None)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.state_dict()

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    # 测试数据集代码: 
    # - gr00t/model/transforms.py 
    #       def collate(features: List[dict], eagle_processor) -> dict:
    #           return features
    # import torch.nn as nn
    # from typing import TYPE_CHECKING, Any, Callable, Optional, Union
    # def training_step(
    #     self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    # ) -> torch.Tensor:
    #     return torch.tensor(1.0, device=model.device)
