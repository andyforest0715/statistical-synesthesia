import os
import json
import time
from openai import OpenAI
#import httpx

# ========== 配置 ==========
INPUT_DIR = ""
OUTPUT_DIR = ""
DEEPSEEK_API_KEY = ""

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 提示词（在这里替换） ==========
SYSTEM_PROMPT = "You are a professional music prompt creator. Output ONLY the three music descriptions. No explanations, no labels, no extra text."

USER_PROMPT_TEMPLATE = """I need you to create three music descriptions optimized for MusicGen based on the video below.
    
    MusicGen Requirements:
    - Use clear genre names and common instrument terms
    - Include tempo/rhythm characteristics AND dynamic changes
    - Describe musical progression (intro→build→climax→outro elements)
    - Include at least 4-5 different instruments/sounds per description
    - Mention rhythmic variations and structural transitions
    - Keep each description 40-50 words
    - Focus on layering, dynamics, and temporal development
    
    Video Description:
    {video_description}
    
    Create three versions:
    
    1. Traditional Match: Expected genre with rich instrumentation and clear progression
    2. Emotional Alternative: Different genre with dynamic arrangement capturing mood evolution
    3. Creative Blend: Genre fusion featuring diverse instruments and structural variety
    
    Format: Output three standalone descriptions, each on a new line.
    
    Example quality: "Upbeat salsa opening with soft piano montunos and shaker, building with brass section entrance, congas joining at chorus. Trumpet solo over full ensemble, finishing with energetic horn stabs and cowbell accents."
    
    Output format: Three lines, no numbering, no labels.
    """
# ========== 主流程 ==========
json_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.json')])
print(f"共 {len(json_files)} 个文件，输出到 {OUTPUT_DIR}")

for idx, json_file in enumerate(json_files):
    video_id = json_file.replace('.json', '')
    out_path = os.path.join(OUTPUT_DIR, f"{video_id}.txt")

    if os.path.exists(out_path):
        continue

    # 读取JSON
    with open(os.path.join(INPUT_DIR, json_file), 'r', encoding='utf-8') as f:
        data = json.load(f)
    video_description = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else str(data)

    # 调用API
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(video_description=video_description)}
            ],
            stream=False
        )
        result = response.choices[0].message.content.strip()
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"[{idx+1}/{len(json_files)}] ✓ {video_id}")
    except Exception as e:
        print(f"[{idx+1}/{len(json_files)}] ✗ {video_id}: {e}")

    time.sleep(0.5)

print("完成")
