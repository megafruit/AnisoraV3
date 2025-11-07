import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance

import torch
from torch.amp import autocast
import torch.nn.functional as F

from network.line_extractor import LineExtractor


# ---------------- utils ----------------

def is_file(path):
    return os.path.splitext(os.path.basename(path))[1]  # TODO: a better way to check if path is file


def is_image(path):
    fname = os.path.basename(path)
    return os.path.splitext(fname)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.webp', '.avif', '.tga']


def is_video(path):
    fname = os.path.basename(path)
    return os.path.splitext(fname)[1].lower() in ['.mp4', '.avi', '.mkv', '.mov']


def increase_sharpness(img, factor=6.0):
    image = Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(image)
    return np.array(enhancer.enhance(factor))


def pil_lanczos_resize(img_pil: Image.Image, size_hw):
    """size_hw = (h, w)  or (w, h)? -> PIL需要(w, h)"""
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    return img_pil.resize(size_hw, resample=resample)


def cv2_area_resize(img_np: np.ndarray, out_wh):
    """OpenCV INTER_AREA：降采样更平滑抗混叠"""
    return cv2.resize(img_np, out_wh, interpolation=cv2.INTER_AREA)


def parse_size(s: str):
    """'1280x720' -> (1280,720)；空字符串->None"""
    if not s or s.lower() in ['none', '']:
        return None
    if 'x' in s.lower():
        w, h = s.lower().split('x')
        return (int(w), int(h))
    raise ValueError("Size format must be like '1280x720'")


# ---------------- model ----------------

def load_model(args):
    if args.mode == 'basic':
        model = LineExtractor(3, 1, True).to(args.device)
    elif args.mode == 'detail':
        model = LineExtractor(2, 1, True).to(args.device)
    else:
        raise ValueError('Mode must be either basic or detail')

    path_model = os.path.join('AniLines-Anime-Lineart-Extractor/weights', f'{args.mode}.pth')
    model.load_state_dict(torch.load(path_model, map_location=torch.device(args.device), weights_only=True))

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


# ---------------- pipeline ----------------

def process_image(path_in, path_out, **kwargs):
    img = cv2.cvtColor(np.array(Image.open(path_in)), cv2.COLOR_RGB2BGR)
    img = inference(img, **kwargs)
    Image.fromarray(img).save(path_out)
    return img


def process_video(path_in, path_out, fourcc='mp4v', **kwargs):
    video = cv2.VideoCapture(path_in)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    video_out = cv2.VideoWriter(path_out, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc='Processing Video'):
        ret, frame = video.read()
        if not ret:
            break
        img = inference(frame, **kwargs)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        video_out.write(img)

    video.release()
    video_out.release()


def upscale_if_needed(bgr_img: np.ndarray, upscale_long: int):
    """把输入按长边放大到 upscale_long（Lanczos），返回 BGR numpy。"""
    if not upscale_long or upscale_long <= 0:
        return bgr_img
    h, w = bgr_img.shape[:2]
    long = max(w, h)
    if long >= upscale_long:
        return bgr_img
    scale = upscale_long / long
    out_w, out_h = int(round(w * scale)), int(round(h * scale))
    # 用 PIL 的 Lanczos（更锐/振铃低）
    up = pil_lanczos_resize(Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)), (out_w, out_h))
    return cv2.cvtColor(np.array(up), cv2.COLOR_RGB2BGR)


def inference(img: np.ndarray, model, args):
    """
    新增逻辑：
      - 进入模型前先按长边放大到 args.upscale_long（Lanczos）
      - 模型输出后，先在灰度阶段缩到 args.final_size（INTER_AREA）
      - 再按 args.binarize / args.binarize_stage 二值化，保证顺滑
    """
    # ---- 1) 可选上采样（Lanczos） ----
    img = upscale_if_needed(img, args.upscale_long)

    # ---- 2) 组装模型输入 ----
    if args.mode == 'basic':
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = increase_sharpness(rgb)
        x_in = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(args.device) / 255.
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = 255 - cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        img_t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float().to(args.device) / 255.
        sobel_t = torch.from_numpy(sobel).unsqueeze(0).unsqueeze(0).float().to(args.device) / 255.
        x_in = torch.cat([img_t, sobel_t], dim=1)

    # ---- 3) padding 到8倍数 ----
    B, C, H, W = x_in.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    x_in = F.pad(x_in, (0, pad_w, 0, pad_h), mode='reflect')

    # ---- 4) 模型推理（fp16可选）----
    with torch.no_grad(), autocast(enabled=args.fp16, device_type='cuda'):
        pred = model(x_in)  # [1,1,H',W'] float 0~1
    pred = pred[:, :, :H, :W]  # 去 padding
    pred_np = pred[0, 0].detach().float().cpu().numpy()  # float [0,1]

    # ---- 5) 降采样到 final_size（在灰度阶段进行，保证顺滑）----
    if args.final_size is not None:
        out_w, out_h = args.final_size
        # OpenCV 期望 (width,height)
        pred_np = cv2_area_resize(pred_np, (out_w, out_h))

    # ---- 6) 可选二值化（仅在 args.binarize∈[0,1] 时生效）----
    if 0.0 <= args.binarize <= 1.0:
        pred_np = (pred_np > args.binarize).astype(np.float32)
    # 输出 uint8 黑白（或灰度）图
    out = np.clip(pred_np * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def do_inference(path_in, path_out, model, args):
    fname = os.path.basename(path_in)
    if is_image(fname):
        process_image(path_in, path_out, model=model, args=args)
    elif is_video(fname):
        process_video(path_in, path_out, fourcc='mp4v', model=model, args=args)
    else:
        raise ValueError(f'Unsupported file: {path_in}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', type=str, default='./snap_frame', help='input directory or file')
    parser.add_argument('--dir_out', type=str, default='./final_output', help='output directory or file')
    parser.add_argument('--mode', type=str, default='detail', help='basic or detail')
    parser.add_argument('--fp16', type=bool, default=True, help='use mixed precision to speed up')
    parser.add_argument('--binarize', type=float, default=-1, help='set to [0, 1] to binarize the output')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')

    # 新增参数
    parser.add_argument('--upscale_long', type=int, default=2048,
                        help='upscale the LONGER side to this size before inference (0 to disable)')
    parser.add_argument('--final_size', type=str, default='1280x720',
                        help="final output size like '1280x720'; empty to keep model size")

    args = parser.parse_args()
    args.final_size = parse_size(args.final_size)

    model = load_model(args)
    flist = [args.dir_in] if os.path.isfile(args.dir_in) else os.listdir(args.dir_in)

    for filename in tqdm(flist, desc='Processing'):
        path_in = filename if is_file(args.dir_in) else os.path.join(args.dir_in, filename)
        path_out = args.dir_out if is_file(args.dir_out) else os.path.join(args.dir_out, os.path.basename(filename))
        if not is_file(args.dir_out):
            os.makedirs(args.dir_out, exist_ok=True)
        do_inference(path_in, path_out, model, args)


if __name__ == '__main__':
    main()
