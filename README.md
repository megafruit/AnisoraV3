# README (English Version)
⚠ I tried ChatGPT to help me directly translate the Chinese version to English one. But after I reading the English version, I
was incredibly envious of ChatGPT's features. It retained my original meaning while refining the structure so much that I had no idea 
how to modify this version. So, if you want to read the README written by me, I suggest you visit the Chinese version. The advantage is 
that you can truly understand my thought process, but the disadvantage is that the structure is simple, which will reduce the reading comfort.

Hello!  
This document contains detailed environment setup instructions and a complete introduction to the workflow of this project.  
If you encounter any issues or have questions about the configuration, please contact: **hxia0469@uni.sydney.edu.au**

---

# Environment Setup

## Stable Diffusion WebUI Setup

### 1. Overview
Stable Diffusion WebUI (hereinafter “WebUI”) is an integrated open-source platform that uses multiple APIs (Stable Diffusion, ControlNet, LoRA, etc.) to generate AI images.  
In this project, WebUI generates the first frame, which is used as the reference frame for Anisora.  
You must install WebUI locally or on a cloud server by yourself.  
WebUI and Anisora do **not** need to share the same directory or environment.

---

### 2. Download Options

This project does not require extremely high GPU performance. Our team ran it on an RTX 4090.  
Better GPUs simply accelerate the generation process.

Two installation options are available:

---

#### **Option A: Baidu Cloud Package (Recommended for NVIDIA GPU users)**  
This package includes the core WebUI components, necessary models, Chinese localization patch, and gallery browser.  
⚠ It can only be used on machines with **NVIDIA GPUs**.

File: **sd.webui.zip**  
Link: https://pan.baidu.com/s/1czxmteAh9Dc0EVagOKa72g?pwd=p8dp  
Extraction code: **p8dp**

---

#### **Option B: Official GitHub Repository**  
If the Baidu package fails to run, or your PC does not use an NVIDIA GPU, or WebUI cannot be installed due to other issues, use the official repository:

Repository: https://github.com/AUTOMATIC1111/stable-diffusion-webui  
Installation guide:  
https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs

---

### 3. Required Extensions and Models

After setting up WebUI, download the required extensions and models.

**Please download all extensions shown checked in the screenshot** (we are unsure which ones were absolutely necessary, so download them all for safety).  
Install via:

**WebUI → Extensions → Load From → search → install**

<p align="center">
  <img src="assets/1.png" width="900" />
</p>

---

### Required Stable Diffusion Models  
Place in:

```
./webui/models/Stable-diffusion
```

1. **AnythingXL_xl.safetensors**  
   https://civitai.com/models/9409?modelVersionId=384264

2. **realcartoonXL_v7.safetensors**  
   https://civitai.com/models/125907/realcartoon-xl

---

### Required LoRA  
Place in:

```
./webui/models/Lora
```

- **LineArtF.safetensors**  
  https://civitai.com/models/596934/line-art-style-sdxl-pony

---

### Required Embeddings  
Place in:

```
./webui/embeddings
```

1. **badhandv4.pt**  
   https://civitai.com/models/16993/badhandv4  

2. **easynegative.safetensors**  
   https://civitai.com/models/7808/easynegative  

---

### Required ControlNet Model  
Place in:

```
./webui/models/ControlNet
```

- **controlnet-canny-sdxl-1.0.safetensors**  
  https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main  

(You only need this file.)

---

⚠ Due to limited project time, we were unable to explore all WebUI features, so the quality of the generated first frame is not perfect.  
You may freely experiment with more models and features to pursue higher-quality first-frame output.

---

# Anisora Setup

## 1. Overview
Anisora is an open-source video-generation system.  
We analyzed it and implemented several modifications based on your input.  
Anisora can generate various anime-style video shots.

---

## 2. Download

Anisora requires powerful GPU memory:

- **1 GPU with ≥80 GB VRAM**, or  
- **Multiple GPUs** (our team used 4 × 48 GB)

Repository:  
https://github.com/megafruit/AnisoraV3.git

Clone:

```bash
git clone https://github.com/megafruit/AnisoraV3.git
```

---

## 3. Environment Setup

Run:

```bash
cd anisoraV3
conda create -n wan_gpu python=3.10
conda activate wan_gpu
pip install -r req-fastvideo.txt
pip install -r requirements.txt
pip install -e .
```

---

## 4. Required Models

### flash-attn (Flash Attention)

Installing via pip often freezes.  
Download manually from:  
https://github.com/Dao-AILab/flash-attention/releases

Use this version:

```
flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Place in project root, then install:

```bash
pip install flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

---

### AnisoraV3 Checkpoints

Download and place into:

```
Index-anisora/V3.1
```

https://huggingface.co/IndexTeam/Index-anisora/tree/main/V3.1

---

### AniLines Pretrained Model

Download from Baidu Cloud and place into:

```
Index-anisora/AniLines-Anime-Lineart-Extractor/weights
```

File: **detail.pth**  
Link: https://pan.baidu.com/s/1KkY_qXgDUM6yA56x5pSITw?pwd=5ph8  
Extraction code: **5ph8**

---

# Running the Pipeline

Refer to the demonstration video for complete results and workflow.  
Below we explain the major commands and parameters.

---

## 0. Initial Step  
Place your natural-language input into:

```
anisoraV3/data/input_txt/input.txt
```

---

## 1. Convert `.txt` → JSON → prompt files

Start with:

```bash
python scene_graph_builder.py
```

This converts your `.txt` file into a structured JSON file.

Then run:

```bash
python read_json.py
```

This generates **three** prompt files:

---

### (1) 1.txt — Prompt for Anisora

Example:

```
At the beginning (first 1.5 seconds), <your scene sentence 1>. 
In the same shot (from 1.5s to 5s), <your scene sentence 2>. @@data/inference-imgs/1.png&&0
```

`@@data/inference-imgs/1.png&&0` defines reference image path and timing.

---

### (2) prompt_for_monochrome_frame.txt — WebUI B&W first frame

Key parameters:

- Model: realcartoonXL_v7.safetensors
- Resolution: 1280×720
- Steps: 20
- CFG: 7

Includes positive & negative prompt templates.

---

### (3) prompt_for_recoloring.txt — WebUI recoloring prompt

Key parameters:

- Model: AnythingXL_xl.safetensors
- Steps: 25
- CFG: 7
- ControlNet canny  
  - weight=1.7  
  - thresholds: Low=100, High=200  
  - Control mode: *My prompt is more important*

Includes templates.

---

## 2. Generate monochrome + recolored first frame using WebUI

Screenshots for reference:

<p align="center">
  <img src="assets/2.png" width="900" />
</p>

<p align="center">
  <img src="assets/3.png" width="900" />
</p>

---

## 3. Generate video using Anisora

Recommended (single GPU):

```bash
python generate-pi-i2v-any.py \
    --task i2v-14B \
    --size 1280*720 \
    --ckpt_dir Index-anisora/V3.1 \
    --image output_videos_any \
    --prompt data/prompt/1.txt \
    --frame_num 81 \
    --sample_steps 6 \
    --sample_shift 8 \
    --sample_guide_scale 1 \
    --use_prompt_extend \
    --prompt_extend_method local_qwen \
    --prompt_extend_target_lang en \
    --prompt_extend_model QwenVL2.5_7B
```

Where,

    --task  
        Please keep the default value: i2v-14B.
    
    --size  
        Can be adjusted as needed, but only specific fixed aspect ratios are supported.
    
    --ckpt_dir  
        Directory where the AnisoraV3 checkpoint models are stored.
    
    --image  
        Directory where the output video frames will be saved.
    
    --prompt  
        Path to the prompt file (1.txt).
    
    --frame_num  
        Number of frames to generate.  
        Since the output video runs at a fixed 16 FPS, this value must be “a multiple of 16 + 1”.
    
    --sample_steps  
        Number of sampling iterations per frame.  
        **Adjust this based on your quality requirements.**
    
    --sample_shift  
        The amount of shift between consecutive frames.  
        Higher values produce stronger visual changes between frames.  
        **Tune this parameter to obtain the best visual quality.**
    
    --sample_guide_scale  
        Controls how strictly the video generation follows the prompt.  
        Higher values enforce closer adherence to the prompt.  
        **Adjust as needed to achieve optimal quality.**
    
    --use_prompt_extend  
    --prompt_extend_method  
    --prompt_extend_target_lang  
    --prompt_extend_model  
        Please keep these four parameters as provided, or remove them entirely.  
        These options enable the QwenVL2.5_7B prompt extension model, which enriches simple prompts and generally improves visual stability and overall video quality.


---

### Multi-GPU Version (4 GPUs)

```bash
torchrun \
    --nproc_per_node=4 \
    --master_port 43210 generate-pi-i2v-any.py \
    --task i2v-14B \
    --size 1280*720  \
    --ckpt_dir Index-anisora/V3.1 \
    --image output_videos_any \
    --prompt data/prompt/1.txt \
    --dit_fsdp --t5_fsdp \
    --ulysses_size 2 \
    --ring_size 2 \
    --frame_num 81 \
    --sample_steps 6 \
    --sample_shift 8 \
    --sample_guide_scale 1 \
    --use_prompt_extend \
    --prompt_extend_method local_qwen \
    --prompt_extend_target_lang en \
    --prompt_extend_model QwenVL2.5_7B
```

---

## 4. Convert colored frames → monochrome lineart

Run:

```bash
python AniLines-Anime-Lineart-Extractor/infer.py
```

The final outputs will appear in:

```
anisoraV3/final_output
```
