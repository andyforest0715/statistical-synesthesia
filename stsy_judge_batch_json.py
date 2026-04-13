"""
===============================================================================
  StSy Video-Music Match — 批量 LLM Judge 评估脚本 (Essentia JSON 版)

  功能: 读取 video JSON + audio JSON → 调用 DeepSeek LLM → 输出 CSV

===============================================================================
"""

import os
import sys
import csv
import json
import time
from pathlib import Path

# =================================================================
#  CONFIG — 只改这里
# =================================================================

VIDEO_DIR  = "/Volumes/T7shield/0-thesis-project/empathybgm/project/videojson"       # 视频描述 JSON 文件夹
AUDIO_DIR  = "/Volumes/T7shield/0-thesis-project/empathybgm/project/metadata"       # Essentia pipeline 输出的 JSON 文件夹
OUTPUT_CSV = "./judge_results_json_project.csv"    # 输出 CSV 路径

# LLM API 配置 (DeepSeek, OpenAI 兼容接口)
#API_KEY    = ""                           # DeepSeek API Key
#BASE_URL   = "https://api.deepseek.com/v1"
#MODEL      = "deepseek-reasoner"

API_KEY    = "sk-RNNuFhIqcfsnSxVdIiH6Lnf7nbjf20ZBs5COsJbRXREjD6VF"
BASE_URL   = "https://api.chatanywhere.tech/v1"
MODEL      = "gemini-3-pro-preview"

# 运行选项
OVERWRITE  = False    # True = 忽略 CSV 中已有的行，全部重跑
DELAY_SEC  = 1.0      # 请求间隔 (避免速率限制)
DRY_RUN    = False    # True = 只打印配对, 不调用 API

# =================================================================


# CSV 列定义
CSV_COLUMNS = [
    "pair",              # 文件名 (不含 .json)
    "match_score",       # 0-10
    "verdict",           # Perfect Match / Strong Match / ...
    "scene_mood_fit",
    "texture_fit",
    "intensity_match",
    "key_mismatch",
    "reasoning",
    "error",             # 失败时的错误信息
]


# =================================================================
#  Prompt — 从 stsy_judge_prompt_json.py 导入
#  确保 stsy_judge_prompt_json.py 与本脚本在同一目录下
# =================================================================

from stsy_judge_prompt_json import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


# =================================================================
#  LLM 调用
# =================================================================

def call_llm(video_text: str, audio_text: str) -> dict:
    """调用 DeepSeek API (OpenAI 兼容接口)，返回解析后的评估 dict。"""
    from openai import OpenAI

    user_msg = USER_PROMPT_TEMPLATE.format(
        video_description=video_text,
        audio_json=audio_text,
    )

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            stream=False,
        )
    except Exception as e:
        return {"match_score": -1, "error": f"API error: {e}"}

    raw = resp.choices[0].message.content.strip()

    # 容忍 markdown 代码块包裹
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        return {"match_score": -1, "error": f"JSON parse failed: {e}"}


# =================================================================
#  CSV 读写
# =================================================================

def load_existing_results(csv_path: str) -> set:
    """读取已有 CSV，返回已完成的 pair 名称集合。"""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("pair"):
                done.add(row["pair"])
    return done


def result_to_row(pair: str, result: dict) -> dict:
    """将 LLM 返回的 dict 展平为 CSV 行。"""
    row = {"pair": pair}
    for col in CSV_COLUMNS[1:]:
        row[col] = result.get(col, "")
    return row


# =================================================================
#  文件配对
# =================================================================

def find_pairs(video_dir: str, audio_dir: str) -> list:
    """按文件名配对 (video=.json, audio=.json)，返回 [(stem, v_path, a_path), ...]。"""
    v_files = {p.stem: p for p in Path(video_dir).glob("*.json")}
    a_files = {p.stem: p for p in Path(audio_dir).glob("*.json")}

    common = sorted(v_files.keys() & a_files.keys())

    v_only = sorted(v_files.keys() - a_files.keys())
    a_only = sorted(a_files.keys() - v_files.keys())
    if v_only:
        print(f"  ⚠ 仅在 video 文件夹: {len(v_only)} 个")
    if a_only:
        print(f"  ⚠ 仅在 audio 文件夹: {len(a_only)} 个")

    return [(stem, str(v_files[stem]), str(a_files[stem])) for stem in common]


# =================================================================
#  主流程
# =================================================================

def main():
    # ── 检查配置 ──
    if not API_KEY and not DRY_RUN:
        print("错误: 未设置 API Key")
        print("  在脚本 CONFIG 区域设置 API_KEY")
        sys.exit(1)

    for d, label in [(VIDEO_DIR, "Video"), (AUDIO_DIR, "Audio")]:
        if not os.path.isdir(d):
            print(f"错误: {label} 目录不存在: {d}")
            sys.exit(1)

    # ── 扫描配对 ──
    print(f"Video: {VIDEO_DIR}")
    print(f"Audio: {AUDIO_DIR}")
    pairs = find_pairs(VIDEO_DIR, AUDIO_DIR)

    if not pairs:
        print("未找到匹配的 JSON 文件对")
        sys.exit(1)

    print(f"找到 {len(pairs)} 对匹配文件\n")

    if DRY_RUN:
        for stem, _, _ in pairs:
            print(f"  {stem}")
        print(f"\n[DRY RUN] 共 {len(pairs)} 对，未调用 API")
        return

    # ── 断点续处理: 读取已有结果 ──
    done = set()
    if not OVERWRITE:
        done = load_existing_results(OUTPUT_CSV)
        if done:
            print(f"CSV 中已有 {len(done)} 条结果，将跳过")

    # ── 打开 CSV (追加模式) ──
    file_exists = os.path.exists(OUTPUT_CSV) and not OVERWRITE
    csv_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)

    if not file_exists or os.path.getsize(OUTPUT_CSV) == 0:
        writer.writeheader()

    # ── 逐对处理 ──
    ok, skip, fail = 0, 0, 0

    try:
        for i, (stem, v_path, a_path) in enumerate(pairs, 1):

            if stem in done:
                print(f"[{i}/{len(pairs)}] {stem} — 已存在，跳过")
                skip += 1
                continue

            print(f"[{i}/{len(pairs)}] {stem}", end="", flush=True)

            # 读取输入
            try:
                with open(v_path, encoding="utf-8") as f:
                    video_data = json.load(f)
                with open(a_path, encoding="utf-8") as f:
                    audio_data = json.load(f)
            except Exception as e:
                print(f" ✗ 读取失败: {e}")
                writer.writerow({"pair": stem, "match_score": -1,
                                 "error": f"File read error: {e}"})
                fail += 1
                continue

            video_text = (json.dumps(video_data, ensure_ascii=False, indent=2)
                          if isinstance(video_data, dict) else str(video_data))
            audio_text = json.dumps(audio_data, ensure_ascii=False, indent=2)

            # 调用 LLM
            print(f" → {MODEL}...", end="", flush=True)
            t0 = time.time()
            result = call_llm(video_text, audio_text)
            elapsed = time.time() - t0

            score = result.get("match_score", -1)
            verdict = result.get("verdict", "")
            error = result.get("error", "")

            if error:
                print(f" ✗ {error[:60]} ({elapsed:.1f}s)")
                fail += 1
            else:
                print(f" → {score}/10 {verdict} ({elapsed:.1f}s)")
                ok += 1

            # 写入 CSV
            writer.writerow(result_to_row(stem, result))
            csv_file.flush()

            if i < len(pairs):
                time.sleep(DELAY_SEC)

    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断，已保存当前进度")
    finally:
        csv_file.close()

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"  完成: {ok} 成功, {skip} 跳过, {fail} 失败")
    print(f"  输出: {os.path.abspath(OUTPUT_CSV)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
