> 本文档记录此代码仓库的环境修改、Bugs修复日志

1. 真机推理依赖库

```bash 
pip install protobuf==5.29.0 # 需要安装此版本,来构建GRPC连接.

Installing collected packages: protobuf
  Attempting uninstall: protobuf
    Found existing installation: protobuf 3.20.3
    Uninstalling protobuf-3.20.3:
      Successfully uninstalled protobuf-3.20.3
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gr00t 1.1.0 requires protobuf==3.20.3, but you have protobuf 5.29.0 which is incompatible.
tensorflow 2.15.0 requires protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3, but you have protobuf 5.29.0 which is incompatible.
```

2. issus: gr00t 微调 libero 出现RAM OOM
- https://github.com/NVIDIA/Isaac-GR00T/issues/119 
    - `I use decord as video backend seem solve the problem`
- https://github.com/NVIDIA/Isaac-GR00T/issues/154 
    - `pip install -U av`

3. **传输指令**
```bash 
# 传输模型至4090
rsync -avz --progress -e "ssh -b 192.168.3.148" Isaac-GR00T jd@192.168.3.140:/home/jd/pi0/
rsync -avz --progress -e "ssh -b 192.168.3.148" resize_img_chunk50_50000steps jd@192.168.3.140:/mnt2/finetune_models/zhaolin/groot/
```

4. 服务器环境中Transformer库与resume model ckpts出现冲突，已解决（修改Transformer库的代码）
```bash 
[rank7]: Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
[rank1]: Traceback (most recent call last):
[rank1]:   File "/export/intent/linzhao/code/Isaac-GR00T/scripts/gr00t_finetune.py", line 293, in <module>
[rank1]:     main(config)
[rank1]:   File "/export/intent/linzhao/code/Isaac-GR00T/scripts/gr00t_finetune.py", line 262, in main
[rank1]:     experiment.train()
[rank1]:   File "/export/intent/linzhao/code/Isaac-GR00T/gr00t/experiment/runner.py", line 171, in train
[rank1]:     self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
[rank1]:   File "/export/intent/linzhao/code/Isaac-GR00T/gr00t/experiment/trainer.py", line 153, in train
[rank1]:     return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
[rank1]:   File "/root/miniforge/envs/groot/lib/python3.10/site-packages/transformers/trainer.py", line 2245, in train
[rank1]:     return inner_training_loop(
[rank1]:   File "/root/miniforge/envs/groot/lib/python3.10/site-packages/transformers/trainer.py", line 2534, in _inner_training_loop
[rank1]:     self._load_rng_state(resume_from_checkpoint)
[rank1]:   File "/root/miniforge/envs/groot/lib/python3.10/site-packages/transformers/trainer.py", line 3130, in _load_rng_state
[rank1]:     checkpoint_rng_state = torch.load(rng_file, weights_only=True)
[rank1]:   File "/root/miniforge/envs/groot/lib/python3.10/site-packages/torch/serialization.py", line 1359, in load
[rank1]:     raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
[rank1]: _pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, [1mdo those steps only if you trust the source of the checkpoint[0m. 
[rank1]: 	(1) Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
[rank1]: 	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
[rank1]: 	WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray._reconstruct was not an allowed global by default. Please use `torch.serialization.add_safe_globals([_reconstruct])` to allowlist this global if you trust this class/function.

修改脚本：/export/intent/linzhao/code/Isaac-GR00T/gr00t/experiment/trainer.py Line 3130

        with safe_globals():
            checkpoint_rng_state = torch.load(rng_file, weights_only=False)

防止resume ckpt报错。 
Isaac-GR00T/ft_resizeimg_chunk.sh # --resume  从5w步恢复训练，训练多任务数据到7w步.
```