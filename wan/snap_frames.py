# snap_frames.py
# -*- coding: utf-8 -*-
"""
每 16 帧截 1 张，从第 17 帧开始（即 17, 33, 49, ...）。
主输出在 OUT_DIR，同时镜像保存到 SECOND_OUT_DIR（若配置）。
并将“本次最后一次截取到的帧”复制到 REF_DIR（/anisoraV3/data/reference-imgs），
文件名从 2.png 起步，如果已存在则依次递增到 3.png、4.png...
"""

import os
import shutil  # 用于复制图片
from typing import List, Optional, Tuple

# ========== Configure ==========
IN_DIR  = "/anisoraV3/output_videos_any"                  # 视频目录（批处理 main() 用）
OUT_DIR = "/anisoraV3/data/snap_frame"                    # 抽帧主输出目录（extract 默认值）
REF_DIR = "../anisoraV3/data/inference-imgs"                # 最后一帧复制到这里（顺位命名从 2.png 开始）

START_AT = 17     # 起始帧（1-based）
STEP = 16         # 抽帧间隔
# ==============================

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


# ----------------- 基础工具 -----------------
def _try_import_cv2():
    """优先尝试 OpenCV（速度快、随机访问友好）"""
    try:
        import cv2
        return cv2
    except Exception:
        return None


def _try_import_imageio():
    """兜底为 imageio（更通用）"""
    try:
        import imageio.v2 as imageio
        return imageio
    except Exception:
        return None


def _ensure_dir(path: Optional[str]):
    """确保目录存在；传 None 或空字符串则忽略"""
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def _list_videos_in_dir(root: str) -> List[str]:
    """列出目录下所有支持的视频文件"""
    vids = []
    if not os.path.isdir(root):
        return vids
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in VIDEO_EXTS:
            vids.append(p)
    return vids


def _every8_indices(total_frames: int) -> List[int]:
    """
    生成 1-based 帧号序列：17, 33, 49, ... <= total_frames
    注意：OpenCV 的定位是 0-based，所以读取时会减 1。
    """
    if total_frames is None or total_frames <= 0:
        return []
    out = []
    i = START_AT
    while i <= total_frames:
        out.append(i)
        i += STEP
    return out

# ----------------- 保存最后一帧为参考图 -----------------
def _save_last_frame_as_reference(saved: List[Tuple[int, str]]):
    """
    将本次保存的最后一帧复制到 REF_DIR，并用 2.png 起步、顺位命名。
    saved: [(frame_idx, saved_path), ...]
    """
    if not saved:
        return
    _ensure_dir(REF_DIR)

    # 取最后一张的文件路径
    last_img_path = saved[-1][1]

    # 从 2.png 开始尝试命名；如果已存在则 +1 继续
    n = 2
    while True:
        candidate = os.path.join(REF_DIR, f"{n}.png")
        if not os.path.exists(candidate):
            break
        n += 1

    shutil.copyfile(last_img_path, candidate)
    print(f"[REF] 参考图已保存：{candidate}")


# ----------------- 抽帧（OpenCV 优先） -----------------
def _extract_every8_cv2(video_path: str, out_dir: str, prefix: Optional[str] = None):
    """
    使用 OpenCV 随机访问指定帧进行保存，速度较快。
    返回：
    {
        "total_frames": int,
        "saved": [(idx, path), ...],
        "out_dir": str
    }
    """
    cv2 = _try_import_cv2()
    if cv2 is None:
        return False  # 交给 imageio 兜底

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _every8_indices(total)
    saved: List[Tuple[int, str]] = []

    for idx in indices:
        pos0 = max(idx - 1, 0)  # OpenCV 用 0-based
        if total > 0 and pos0 >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        base = prefix or os.path.splitext(os.path.basename(video_path))[0]
        file_name = f"{base}_f{idx:06d}.png"
        out_path = os.path.join(out_dir, file_name)
        cv2.imwrite(out_path, frame)
        saved.append((idx, out_path))

    cap.release()
    return {"total_frames": total, "saved": saved, "out_dir": out_dir}


# ----------------- 抽帧（imageio 兜底） -----------------
def _extract_every8_imageio(video_path: str, out_dir: str, prefix: Optional[str] = None):
    """
    兜底方案：不知道总帧数也没关系，顺序遍历并在第 17,33,... 帧处保存。
    返回结构与 OpenCV 版一致。
    """
    imageio = _try_import_imageio()
    if imageio is None:
        raise ImportError(
            "未安装 OpenCV 或 imageio，无法读取视频。请先安装：\n"
            "  pip install opencv-python\n"
            "或\n"
            '  pip install "imageio[ffmpeg]"'
        )

    reader = imageio.get_reader(video_path)
    saved: List[Tuple[int, str]] = []
    base = prefix or os.path.splitext(os.path.basename(video_path))[0]

    try:
        # 优先尝试拿总帧数（如果能拿到就直接算索引）
        try:
            total = reader.get_length()
        except Exception:
            total = None

        if total and total > 0:
            indices = _every8_indices(total)
            for idx in indices:
                pos0 = max(idx - 1, 0)
                try:
                    frame = reader.get_data(pos0)
                except Exception:
                    continue
                file_name = f"{base}_f{idx:06d}.png"
                out_path = os.path.join(out_dir, file_name)
                imageio.imwrite(out_path, frame)
                saved.append((idx, out_path))
        else:
            # 无总帧数：按序遍历，第 17,33,49,... 帧保存
            target = START_AT
            frame_id_1b = 0  # 1-based 计数
            for frame in reader:
                frame_id_1b += 1
                if frame_id_1b == target:
                    file_name = f"{base}_f{frame_id_1b:06d}.png"
                    out_path = os.path.join(out_dir, file_name)
                    imageio.imwrite(out_path, frame)
                    saved.append((frame_id_1b, out_path))
                    target += STEP
    finally:
        try:
            reader.close()
        except Exception:
            pass

    return {"total_frames": total, "saved": saved, "out_dir": out_dir}


# ----------------- 对外主函数（已自动保存最后一帧到 REF_DIR） -----------------
def extract_frames_every8(video_path: str, out_dir: Optional[str] = None, prefix: Optional[str] = None):
    """
    每 16 帧截 1 张，从第 17 帧开始。
    优先 OpenCV，失败回退 imageio。
    额外行为：自动把“本次最后一次截取到的帧”复制到 REF_DIR 并按 2.png 起步顺位命名。
    返回: {"total_frames": int|None, "saved": [(idx, path), ...], "out_dir": str}
    """
    out_dir = out_dir or OUT_DIR
    _ensure_dir(out_dir)

    info = None
    cv2 = _try_import_cv2()
    if cv2 is not None:
        try:
            info = _extract_every8_cv2(video_path, out_dir, prefix)
        except Exception:
            info = None  # 失败则回退 imageio

    if info is None:
        info = _extract_every8_imageio(video_path, out_dir, prefix)

    # --- 新增：无论从哪条路径抽帧，都会在此自动保存“最后一帧”为参考图 ---
    try:
        _save_last_frame_as_reference(info.get("saved", []))
    except Exception as _e:
        # 不让参考图失败影响主流程
        print(f"[WARN] 参考图保存失败：{_e}")

    return info


# ----------------- CLI（批处理整个目录，可不使用） -----------------
def main():
    _ensure_dir(OUT_DIR)
    _ensure_dir(REF_DIR)         # 确保参考图目录存在
    videos = _list_videos_in_dir(IN_DIR)
    if not videos:
        print(f"[INFO] 未在 {IN_DIR} 发现视频文件（支持扩展名：{sorted(VIDEO_EXTS)}）")
        return

    print(f"[INFO] 在 {IN_DIR} 发现 {len(videos)} 个视频")
    print(f"[INFO] 主输出目录：{OUT_DIR}")
    print(f"[INFO] 参考图目录：{REF_DIR}")

    for vp in videos:
        base = os.path.splitext(os.path.basename(vp))[0]
        try:
            info = extract_frames_every8(vp, OUT_DIR, prefix=base)
            saved = info["saved"]
            if saved:
                print(f"[OK] {base}: 保存 {len(saved)} 张 -> {OUT_DIR}")
            else:
                print(f"[WARN] {base}: 未能保存任何帧")
        except Exception as e:
            print(f"[ERR] {base}: {e}")


if __name__ == "__main__":
    main()
