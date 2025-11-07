# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

#------------------------ Wan shared config ------------------------#
wan_shared_cfg = EasyDict()

# t5
wan_shared_cfg.t5_model = 'umt5_xxl'
wan_shared_cfg.t5_dtype = torch.bfloat16
wan_shared_cfg.text_len = 512

# transformer
wan_shared_cfg.param_dtype = torch.bfloat16

# inference
wan_shared_cfg.num_train_timesteps = 1000
wan_shared_cfg.sample_fps = 16
wan_shared_cfg.sample_neg_prompt = '没有清晰的轮廓线，没有黑色轮廓线，背景没有黑色轮廓线，运动模糊，镜头模糊，背景虚化，角色虚化，bilibili水印，狭窄的画面，不够宽广的画面，过于细节，画面太复杂，画面过亮，画面过暗，角色凭空出现，角色凭空消失，缺少提示词描述的角色，色调艳丽，过曝，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，三条腿，静止不动的画面，杂乱的背景，背景人很多，倒着走，亮度波动，曝光波动，色彩闪烁，亮度闪烁，黑场闪烁，高光溢出，局部过曝，局部过暗，色偏，忽明忽暗'
