import os
import torch
import zipfile
import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from openai import OpenAI
import sys
from tqdm import tqdm

sys.path.append(os.path.split(sys.path[0])[0])
from datasets.mp3d import MP3DDataset

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.ohmygpt.com/v1")

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def annotate_video(frames, initial_prompt):
    image_urls = []
    for frame in frames:
        preprocessed_frame = preprocess_image(frame)
        image_urls.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_to_base64(preprocessed_frame)}"
            }
        })
    
    prompt_content = [{"type": "text", "text": initial_prompt}] + image_urls
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_content}],
        max_tokens=256
    )
    
    return response.choices[0].message.content

def preprocess_image(image):
    # Placeholder for actual image preprocessing logic
    return image

def visualize_and_save(frames, description, save_path):
    num_frames = len(frames)
    num_cols = 6
    num_rows = (num_frames + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))
    axes = axes.flatten()
    
    for i, frame in enumerate(frames):
        axes[i].imshow(frame)
        axes[i].axis('off')
    
    # Remove unused axes
    for i in range(num_frames, len(axes)):
        axes[i].axis('off')

    # Add description at the bottom
    plt.figtext(0.5, 0.05, description, wrap=True, horizontalalignment='center', fontsize=12)
    
    plt.tight_layout(pad=0.5, h_pad=0, w_pad=1.0)  # Adjust padding between images
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(save_path)
    plt.close()

class Configs:
    data_path = "data/v1/scans"
    num_frames = 18

if __name__ == '__main__':
    configs = Configs()
    dataset = MP3DDataset(configs=configs, random=False, return_pt=False)

    initial_prompt = "Describe the scene, objects, and relationships in these images in a overall level, keep response short:"

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    for i, data in tqdm(enumerate(dataset)):
        frames = data['frames']
        pano_name = data['pano_name']
        save_path = os.path.join(output_dir, f"{pano_name}_description.png")

        if os.path.exists(save_path):
            continue

        overall_description = annotate_video(frames, initial_prompt)
        print(f"Panorama {pano_name} description:\n {overall_description}")
        print("-" * 30)

        visualize_and_save(frames, overall_description, save_path)
        break
