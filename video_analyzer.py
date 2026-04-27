import os
import json
import time
from datetime import datetime
import torch
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from eval.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

# ========== 配置 ==========
VIDEO_DIR = "/root/autodl-tmp/try"  # 视频文件夹
OUTPUT_DIR = "/root/tryanalysis"            # 输出文件夹
MODEL_PATH = "/root/autodl-tmp/models--wyccccc--TimeChatOnline-7B/snapshots/4aa3f6f531214f0019c86c6c8e0a16d98af5d46a"

DROP_METHOD = 'feature'
DROP_THRESHOLD = 0.5
DROP_ABSOLUTE = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 加载模型（只加载一次）==========
print("正在加载模型...")

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    quantization_config=quantization_config,
    device_map="auto",
)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto",
# )
processor = Qwen2VLProcessor.from_pretrained(MODEL_PATH)
print("✓ 模型加载完成")

# questions = [
#     {"key": "description",
#      "prompt": "Describe this video."},
# ]
questions = [
        {
            "title": "CONTENT & NARRATIVE",
            "prompt": "Describe what happens in this video from beginning to end. Focus on the actions, story progression, and what is being demonstrated or advertised."
        },
        {
            "title": "VISUAL TECHNIQUES", 
            "prompt": "Without describing the actions again, analyze ONLY the visual techniques: What colors dominate each scene? Is the lighting bright or moody? Are the shots wide or close-up? Is the camera static or moving? How fast is the editing? What visual effects are used?"
        },
        {
            "title": "EMOTIONAL JOURNEY",
            "prompt": "Describe in detail how this video makes you feel as you watch it. What is the initial mood? How does the energy level change throughout? What emotions does the music and pacing evoke? Which moments feel most intense or relaxed? How does the video want the viewer to feel about the product?"
        }
    ]
# ========== 分析单个视频 ==========
def analyze_video(video_path):
    results = {}
    for q in questions:
        # 列表格式 —— 给 process_vision_info 用
        messages = [{"role": "user", "content": [
            {"type": "video", "video": "file://"+video_path, "max_frames": 32, "max_pixels": 336*336},
            {"type": "text", "text": q["prompt"]},
        ]}]
        image_inputs, video_inputs = process_vision_info(messages)

        # 字符串格式 —— 给 apply_chat_template 用
        video_token = "<|vision_start|><|video_pad|><|vision_end|>"
        text_messages = [{"role": "user", "content": video_token + q["prompt"]}]
        text = processor.apply_chat_template(text_messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=1500,
                drop_method=DROP_METHOD, drop_threshold=DROP_THRESHOLD, drop_absolute=DROP_ABSOLUTE,
                dr_save_path=f"drop_{curr_time}.jsonl",
            )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
        output = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        results[q["title"]] = output
        del inputs, generated_ids, trimmed, image_inputs, video_inputs
        torch.cuda.empty_cache()
    return results

# ========== 批量处理 ==========
video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))])
print(f"共 {len(video_files)} 个视频")

for idx, vf in enumerate(video_files):
    name = os.path.splitext(vf)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{name}.json")

    if os.path.exists(out_path):
        continue

    print(f"\n[{idx+1}/{len(video_files)}] {name}")
    try:
        results = analyze_video(os.path.join(VIDEO_DIR, vf))
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 完成")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ✗ 失败: {e}")

print("\n全部完成")
