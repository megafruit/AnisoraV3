# -*- coding: utf-8 -*-
"""
read_json.py (no-arg runner)

放置位置：anisoraV3/read_json.py

功能：
1) 先清空以下目录（保留目录本身）：
   - output_videos_any
   - snao_frame  (按你提供的名字)
   - snap_frame  (容错：常见拼写)
   - data/inference-imgs   （保留 1.png）
   - data/prompt
2) 读取 data/input_json/*.json
   （已实现）为每个 JSON 抽取 T1 的 scene，生成 WebUI 正/负提示词（线稿单色版）汇总到 data/prompt/prompt_for_monochrome_frame.txt
   （新增）再次基于 T1 生成你指定“simple style 版”的正/负提示词，汇总到 data/prompt/prompt_for_simple_style.txt
3) 将结果依次写入 data/prompt/1.txt, 2.txt, ... （多个 JSON 会连续编号）
   - 若某 JSON 只有 T1，则只输出一条（单句格式）
   - 否则输出 (n-1) 条（两句格式）
4) 不需要任何命令行参数
"""

import os
import re
import json
import shutil
from typing import List, Dict, Any, Tuple

# ---------- 基础工具 ----------
def ensure_dir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def clear_directory(path: str) -> int:
    """
    清空目录的所有内容（文件与子目录），保留目录本身。返回删除条目数量。
    """
    ensure_dir(path)
    removed = 0
    for entry in os.scandir(path):
        try:
            if entry.is_file() or entry.is_symlink():
                os.remove(entry.path)
                removed += 1
            elif entry.is_dir():
                shutil.rmtree(entry.path)
                removed += 1
        except Exception as e:
            print(f"[WARN] 无法删除 {entry.path}: {e}")
    return removed

def clear_directory_keep_file(path: str, keep_name: str) -> int:
    """
    清空目录内容但保留某个特定文件 (按文件名匹配)，
    目录本身保留。返回删除条目数量。
    """
    ensure_dir(path)
    removed = 0
    for entry in os.scandir(path):
        basename = os.path.basename(entry.path)
        if basename == keep_name:
            continue
        try:
            if entry.is_file() or entry.is_symlink():
                os.remove(entry.path)
                removed += 1
            elif entry.is_dir():
                shutil.rmtree(entry.path)
                removed += 1
        except Exception as e:
            print(f"[WARN] 无法删除 {entry.path}: {e}")
    return removed

# ---------- T 序排序与 scene 清理 ----------
TIME_RE = re.compile(r'^\s*[Tt]\s*(\d+)\s*$')  # 匹配 "T1" / "t 2" / " T3 " 等

def parse_time_order(item: Dict[str, Any], fallback_idx: int) -> Tuple[int, int]:
    """
    返回排序键 (t_number, fallback_idx)
    - t_number 为从 time 字段解析出的数字（解析失败则用极大值）
    - fallback_idx 用于稳定排序
    """
    t = str(item.get("time", "")).strip()
    m = TIME_RE.match(t)
    if m:
        try:
            return (int(m.group(1)), fallback_idx)
        except Exception:
            pass
    return (10**9, fallback_idx)

def clean_scene_text_for_sentence(s: str) -> str:
    """
    清理 scene 文本（用于句子拼接）：
    - 去掉首尾空白
    - 将内部多空格压缩为单空格
    - 去掉末尾多余的逗号/句点/分号/空白，并统一补一个句点
    """
    if s is None:
        s = ""
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[\s\.,;，；。]+$', '', s)
    if s and not s.endswith('.'):
        s += '.'
    return s

def clean_scene_text_for_prompt(s: str) -> str:
    """
    清理 scene 文本（用于 WebUI 提示词）：
    - 去掉首尾空白
    - 压缩多空格
    - 去掉结尾的标点，不再补句号（避免把句号带入 prompt）
    """
    if s is None:
        s = ""
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[\s\.,;，；。]+$', '', s)
    return s

# ---------- 文本模板 ----------
def build_prompt_two_scenes(scene_a: str, scene_b: str, i: int) -> str:
    scene_a = clean_scene_text_for_sentence(scene_a)
    scene_b = clean_scene_text_for_sentence(scene_b)
    return (
        f"At the beginning (first 1.5 seconds), {scene_a} "
        f"In the same shot (from 1.5s to 5s), {scene_b} "
        f"@@data/inference-imgs/{i}.png&&0"
    )

def build_prompt_single_scene(scene_a: str, i: int = 1) -> str:
    scene_a = clean_scene_text_for_sentence(scene_a)
    return (
        f"At the beginning (first 2 seconds), {scene_a} "
        f"@@data/inference-imgs/{i}.png&&0"
    )

# ---- 线稿/单色版 提示词（已存在）----
def build_webui_prompts_monochrome(scene1: str) -> Tuple[str, str]:
    s = clean_scene_text_for_prompt(scene1)
    positive = (
        f"({s}:1.3), black-and-white comic, (cute, cartoon:1.2), "
        f"clean outline, sharp edges, (lineart, monochrome:1.4), "
        f"correct proportions, no overlapping, (masterpiece, best quality:1.7) "
        f"<lora:LineArtF:0.95>"
    )
    negative = (
        "duplicate humans, duplicate animals, wrong number of humans, "
        "wrong number of animals, furry, wrong species:1.9, "
        "(worst face, mutilated face, no eyeball:1.6), "
        "hybrid creatures, fused species, bad anatomy, "
        "(extra limbs:1.7), duplicate human, detailed background, "
        "wrong gender, badhandv4, easynegative"
    )
    return positive, negative

# ---- 新增：simple style 版 提示词（按你给定格式）----
def build_webui_prompts_simple(scene1: str) -> Tuple[str, str]:
    s = clean_scene_text_for_prompt(scene1)
    positive = (
        f"({s}:0.8), simple style, (simple background:1.2), "
        f"(clear outline, closed line, clear eyes, best face, best quality:1.5)"
    )
    negative = (
        "no color, (bad eyes, no eyeballs, poorly drawn face, rough face, extra limbs:1.5), "
        "(badhandv4, easynegative:1.5)"
    )
    return positive, negative

def read_json_file(fp: str) -> List[Dict[str, Any]]:
    with open(fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{fp} 的顶层结构不是数组 list。")
    return data

def pick_T1_scene(items: List[Dict[str, Any]]) -> str:
    """
    从 items 中优先找 time == T1 的 scene；若不存在则取排序后的第一个 scene。
    """
    for it in items:
        t = str(it.get("time", "")).strip()
        m = TIME_RE.match(t)
        if m:
            try:
                if int(m.group(1)) == 1:
                    return str(it.get("scene", "") or "")
            except Exception:
                pass
    return str(items[0].get("scene", "") or "")

# ---------- 主流程 ----------
def main():
    # 以本文件所在目录作为项目根（要求把本文件放在 anisoraV3/ 下）
    ROOT = os.path.dirname(os.path.abspath(__file__))

    # 目录路径
    DIR_OUTPUT_VIDS   = os.path.join(ROOT, "output_videos_any")
    DIR_SNAP_FRAME    = os.path.join(ROOT, "snap_frame")      # 容错：常见拼写
    DIR_INFER_IMGS    = os.path.join(ROOT, "data", "inference-imgs")
    DIR_PROMPT_OUT    = os.path.join(ROOT, "data", "prompt")
    DIR_JSON_INPUT    = os.path.join(ROOT, "data", "input_json")
    DIR_FINAL_OUTPUT  = os.path.join(ROOT, "final_output")

    print("[INFO] 清理目录内容（保留目录本身）:")

    # 这些目录直接清空
    for p in [DIR_OUTPUT_VIDS, DIR_SNAP_FRAME, DIR_PROMPT_OUT, DIR_FINAL_OUTPUT]:
        removed = clear_directory(p)
        print(f"  - {p}  删除 {removed} 项")

    # inference-imgs 目录：清空但保留 1.png
    removed_imgs = clear_directory_keep_file(DIR_INFER_IMGS, keep_name="1.png")
    print(f"  - {DIR_INFER_IMGS}  删除 {removed_imgs} 项 (保留 1.png)")

    # 读取 json_input
    if not os.path.isdir(DIR_JSON_INPUT):
        raise FileNotFoundError(f"找不到输入目录：{DIR_JSON_INPUT}")

    ensure_dir(DIR_PROMPT_OUT)

    files = [
        os.path.join(DIR_JSON_INPUT, fn)
        for fn in os.listdir(DIR_JSON_INPUT)
        if fn.lower().endswith(".json")
    ]
    if not files:
        print(f"[INFO] {DIR_JSON_INPUT} 中未发现 .json 文件。")
        return

    files.sort()
    global_idx = 1  # 跨文件连续编号，避免覆盖

    # 汇总文件：线稿单色版
    mono_fp = os.path.join(DIR_PROMPT_OUT, "prompt_for_monochrome_frame.txt")
    with open(mono_fp, "w", encoding="utf-8") as wf:
        wf.write("## 此提示词用于绘制单色的第一帧\n重要参数:\n1. Stable Diffusion模型选择: realcartoonXL_v7.safetensors\n2. 分辨率: 1280*720\n3. Sampling steps = 20\n4. CFG = 7\n\n")

    # 新增汇总文件：simple style 版
    simple_fp = os.path.join(DIR_PROMPT_OUT, "prompt_for_recoloring.txt")
    with open(simple_fp, "w", encoding="utf-8") as wf:
        wf.write("## 此提示词用于为单色的第一帧上色\n重要参数:\n1. Stable Diffusion模型选择: AnythingXL_xl.safetensorsn\n2. 分辨率: 1280*720\n3. Sampling steps = 25\n4. CFG = 7\n"
                 "5. ControlNet: canny算法, weight=1.7, Starting Control Step=0.0, Ending Control Step=1.0, Control mode=My prompt is more important, Low threshold = 100, High threshold = 200\n\n")

    for fp in files:
        try:
            data = read_json_file(fp)

            # 根据 time(T1,T2,...) 排序
            ordered = sorted(
                enumerate(data),
                key=lambda pair: parse_time_order(pair[1], pair[0])
            )
            items = [item for _, item in ordered]

            # 提取 scene 列表
            scenes = [str(item.get("scene", "") or "") for item in items]
            n = len(scenes)

            if n <= 0:
                print(f"[WARN] {fp} 为空，跳过。")
                continue

            # 选出 T1 的 <scene1>
            scene1 = pick_T1_scene(items)

            # === 线稿单色版 ===
            pos_mono, neg_mono = build_webui_prompts_monochrome(scene1)
            with open(mono_fp, "a", encoding="utf-8") as wf:
                wf.write(f"正面提示词: {pos_mono}\n")
                wf.write(f"负面提示词: {neg_mono}\n\n")

            # === 新增：simple style 版 ===
            pos_simple, neg_simple = build_webui_prompts_simple(scene1)
            with open(simple_fp, "a", encoding="utf-8") as wf:
                wf.write(f"正面提示词: {pos_simple}\n")
                wf.write(f"负面提示词: {neg_simple}\n\n")

            # 只有一个场景 → 单句模板
            if n == 1:
                content = build_prompt_single_scene(scenes[0], i=global_idx)
                out_fp = os.path.join(DIR_PROMPT_OUT, f"{global_idx}.txt")
                with open(out_fp, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"[OK] {fp} -> {out_fp}")
                global_idx += 1
                continue

            # 多个场景 → 相邻两两组合
            for i in range(1, n):
                scene_a = scenes[i - 1]
                scene_b = scenes[i]
                content = build_prompt_two_scenes(scene_a, scene_b, i=global_idx)
                out_fp = os.path.join(DIR_PROMPT_OUT, f"{global_idx}.txt")
                with open(out_fp, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"[OK] {fp} -> {out_fp}")
                global_idx += 1

        except Exception as e:
            print(f"[ERROR] 处理 {fp} 失败：{e}")

    print(f"[DONE] 共输出 {global_idx - 1} 个 .txt 到 {DIR_PROMPT_OUT}")
    print(f"[DONE] 已生成线稿单色提示词汇总：{mono_fp}")
    print(f"[DONE] 已生成 Simple Style 提示词汇总：{simple_fp}")

if __name__ == "__main__":
    main()
