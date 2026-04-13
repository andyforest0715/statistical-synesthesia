import os
import scipy.io.wavfile
from transformers import pipeline

# ========== 配置 ==========
INPUT_DIR = "/root/tryanalysis"   # txt提示词文件夹
OUTPUT_DIR = "/root/tryanalysi2"        # 输出wav文件夹
MODEL = "/root/autodl-tmp/models--facebook--musicgen-large/snapshots/15ccdc92099879e47b6da12c350cdb71d4eab3ca"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型（只加载一次）
print("正在加载MusicGen-large...")
synthesiser = pipeline("text-to-audio", model=MODEL, device=0)
print("✓ 模型加载完成")

# 批量处理
txt_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')])
print(f"共 {len(txt_files)} 个提示词文件")

for idx, txt_file in enumerate(txt_files):
    name = txt_file.replace('.txt', '')
    out_path = os.path.join(OUTPUT_DIR, f"{name}.wav")

    if os.path.exists(out_path):
        continue

    with open(os.path.join(INPUT_DIR, txt_file), 'r', encoding='utf-8') as f:
        prompt = f.read().strip().split('\n')[1]  # 取第一行

    if len(prompt) > 400:
        prompt = prompt[:400]

    try:
        music = synthesiser(prompt, forward_params={"do_sample": True})
        scipy.io.wavfile.write(out_path, rate=music["sampling_rate"], data=music["audio"])
        print(f"[{idx+1}/{len(txt_files)}] ✓ {name}")
    except Exception as e:
        print(f"[{idx+1}/{len(txt_files)}] ✗ {name}: {e}")

print("完成")
