import base64
import os
import requests
from pathlib import Path
from typing import List, Tuple, Dict, Any
import cv2
import json
import concurrent.futures

"""
脚本: python data_process/aug_instruction.py
将数据集中的指令进行改写
"""

os.environ["OPENAI_API_KEY"] = "bfa57f82-41fc-49a7-aadf-edc937df3f97"
os.environ["OPENAI_API_BASE"] = "http://gpt-proxy.jd.com/v1/chat/completions"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# SYSTEM_PROMPT = """\
# You are an AI assistant specialized in generating robot operation instructions from video frames. Your task is to analyze the input video frames and output a concise instruction string in the format: "pick up the {object name} on the {color name} region".
#
# **Processing Steps**:
# 1. **Identify the target object**:
#    - Determine the primary object that the robotic arm is directly targeting.
# 2. **Identify the associated colored region**:
#    - Locate the colored region.
#    - Extract the color name as a simple descriptor (e.g., "red" or "green"; avoid complex hues).
# 3. **Format the instruction**:
#    - Insert the identified object name into "{object name}" and color name into "{color name}".
#    - Output only the string in the exact format: "pick up the [object name] on the [color name] region". Do not add any explanations, prefixes, or extra characters.
#    - If the input frame lacks a clear object or color region, default to "pick up the object on the region" but only as a fallback (ideally, frame descriptions should provide enough detail).
#
# **Output Constraints**:
# - Output must be a single string with no additional text.
# - Keep object names simple (e.g., "plastic bottle", not "brown container").
# - Use common color names (e.g., "red", "green", "blue") based on dominant hue.
# - Ensure the instruction reflects the scene's immediate action (e.g., robotic arm approaching an object).
# """

"""
说明: 文本指令采用模板化的方式进行生成：
"pick up the {object name} on the {color name} region"
"""


# SYSTEM_PROMPT = """\
# You are an AI assistant specialized in generating robot operation instructions from video frames. Your task is to analyze the input video frames and output a concise instruction string **exclusively** in the format: "Pick up the {object name} from the table. Place the {object name} in the {color name} area.".
#
# **Processing Steps**:
# 1. **Identify the target object**:
#    - Determine the primary object that the robotic arm is directly targeting.
#    - **Critical Constraint**: Object must be identified **ONLY** as:
#      - `bottle` (if container has narrow neck/cap, e.g., plastic bottle, glass bottle)
#      - `cup` (if open-topped container, e.g., paper cup, ceramic cup)
#      - `fruit`: e.g. apple, banana
#      *Ignore all other objects*.
#
# 2. **Identify the associated colored region**:
#    - Locate the dominant colored region directly beneath or adjacent to the target object.
#    - Extract the color name using **simple descriptors** (e.g., "red", "green", "blue").
#
# 3. **Format the instruction**:
#    - Insert `bottle` or `cup` into "{object name}" and color into "{color name}".
#    - **Strict Output Format**: "pick up the [bottle/cup] on the [color] region"
#      *Example: "pick up the bottle on the red region"*
#    - **DO NOT** add explanations, prefixes, or extra characters.
#    - **Fallback** (only if no cup/bottle): "pick up the object on the region"
#
# **Output Rules**:
# 1. Output must be a **single string** with no additional text.
# 2. **Object names must be**:
#    - `bottle` for any bottle-like containers
#    - `cup` for any cup-like containers
#    *(Never use descriptors like "plastic", "glass", etc.)*
# 3. Colors use **basic names** (e.g., red, not crimson/scarlet).
# 4. Instruction must reflect the robot's immediate action.
# """

SYSTEM_PROMPT = """\
You are an expert in generating robot operation instructions from video frames. 
Your task is to analyze the input video frames and output a single concise instruction in the following strict format:
- "Pick up the {object name} and place it in the {color name} area."

**Instructions**:

**Object Identification**:
Identify the main object being manipulated by the robotic arm in the final frame.
Object name: bottle, cola bottle, cup, or other objects.
For unrecognized objects, uniformly use "object" as the description.

**Color Region Identification**:
Determine the dominant color region where the object is placed in the final frame.
Color names: red, blue, green, orange, pink, yellow, brown, or other colors.

**Output Format**:
**Output only**:
"Pick up the {object name} and place it in the {color name} area."
**Rules**:
Output only one instruction, with no explanations or extra text.
If no valid object is present is present, output:
"Pick up the object and place it in the {color name} area."
Do not add any explanations, prefixes, or extra content.
"""

class OpenAIModel:
    def __init__(self, base_url: str=os.environ["OPENAI_API_BASE"], 
                 api_key: str=os.environ["OPENAI_API_KEY"], 
                 model: str="gpt-4o-0806", temperature: float=0.1, max_tokens: int=4096,
                 system_prompt: str=SYSTEM_PROMPT):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def get_model_response(self, prompt: str=None, images: List[str]=[]) -> Tuple[bool, str]:
        content = []
        if prompt != None:
            if "gpt" in self.model:
                content.append(
                    {
                        "type": "text",
                        "text": prompt
                    }
                )
            else:
                content = prompt
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }
                }
            )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        response = requests.post(self.base_url, headers=headers, json=payload).json()
        if "error" not in response:
            return True, response["choices"][0]["message"]["content"]
        else:
            return False, response # response["error"]["message"]


def process_video(video_file: Path, model: OpenAIModel) -> Dict[str, Any]:
    """Process a single video file to extract task_index and generate instruction"""
    try:
        task_index = int(video_file.stem.split('_')[1])
    except (IndexError, ValueError):
        print(f"Skipping invalid filename: {video_file.name}")
        return None

    video_capture = cv2.VideoCapture(str(video_file))
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    base64_images = []

    if frame_count > 0:
        indices = sorted(set([0, frame_count // 2, frame_count - 1])) # frame_count // 3, 2 * frame_count // 3,

        for idx in indices:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = video_capture.read()
            if success:
                _, buffer = cv2.imencode('.jpg', frame)
                base64_image = base64.b64encode(buffer).decode('utf-8')
                base64_images.append(base64_image)

    video_capture.release()

    if base64_images:
        success, response = model.get_model_response(images=base64_images)
        if success:
            return {"task_index": task_index, "task": response.strip()}
        else:
            print(f"Model error for {video_file}: {response}")
    else:
        print(f"Could not extract frames from {video_file}")

    return None


def process_dataset(dataset_path: str, model: OpenAIModel):
    """Process all videos in a dataset directory and save results at once"""
    # Prepare output directory and file
    save_dir = Path(dataset_path) / "meta"
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "new_tasks.jsonl"

    # Get all video files
    video_path = Path(dataset_path) / "videos/chunk-000/observation.images.cam_head_left"
    all_videos = list(video_path.glob("*.mp4"))
    results = []

    # Create thread pool for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all video processing tasks
        future_to_video = {executor.submit(process_video, video, model): video for video in all_videos}

        # Collect results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_video)):
            video_file = future_to_video[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"Processed {video_file.name} [success]")
            except Exception as e:
                print(f"Error processing {video_file.name}: {str(e)}")

            # Print progress every 10 videos
            if (i + 1) % 10 == 0:
                print(f"Progress: {len(results)}/{len(all_videos)} videos processed")

    # Sort results by task_index before saving
    sorted_results = sorted(results, key=lambda x: x["task_index"])

    # Write all results to file at once
    with open(output_path, 'w') as jsonl_file:
        for result in sorted_results:
            jsonl_file.write(json.dumps(result) + '\n')

    print(f"Saved {len(sorted_results)} tasks to {output_path}")
    return sorted_results


if __name__ == "__main__":
    model = OpenAIModel()

    # 这里修改数据集路径,会对前5天抓杯子的数据进行instructions改写
    dataset_paths = [
        "/home/robot/pi/datasets/rm_lerobot/day1",
        "/home/robot/pi/datasets/rm_lerobot/day2",
        "/home/robot/pi/datasets/rm_lerobot/day3",
        "/home/robot/pi/datasets/rm_lerobot/day4",
        "/home/robot/pi/datasets/rm_lerobot/day5",
        "/home/robot/pi/datasets/rm_lerobot/day6",
        "/home/robot/pi/datasets/rm_lerobot/day7",
        "/home/robot/pi/datasets/rm_lerobot/day8",
        "/home/robot/pi/datasets/rm_lerobot/day9",
        "/home/robot/pi/datasets/rm_lerobot/day10",
    ]

    # Process each dataset sequentially
    for dataset_path in dataset_paths:
        print(f"\n{'=' * 50}")
        print(f"Starting processing for: {dataset_path}")
        print(f"{'=' * 50}\n")

        results = process_dataset(dataset_path, model)

        print(f"\n{'=' * 50}")
        print(f"Completed processing for: {dataset_path}")
        print(f"Processed {len(results)} videos")
        print(f"{'=' * 50}\n")