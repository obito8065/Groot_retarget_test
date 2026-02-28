import torch
import torchvision.transforms.v2 as T
from PIL import Image
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import build_eagle_processor
from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH
import numpy as np

@torch.no_grad()
def extra_features(model, input_ids, pixel_values, attention_mask):
    input_embeds = model.language_model.get_input_embeddings()(input_ids)
    vit_embeds = model.extract_feature(pixel_values)

    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    input_ids = input_ids.reshape(B * N)
    selected = input_ids == model.image_token_index

    input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)

    input_embeds = input_embeds.reshape(B, N, C)

    outputs = model.language_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=True,
    )

    return outputs, vit_embeds


def visualize_feature_map(features, original_height, original_width, original_image, ratio=0.5):
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    # 获取特征
    feature_maps = features['features']
    print("Feature maps shape:", feature_maps.shape)

    # 计算预期的patch数量
    input_height, input_width = 224, 224  # 处理器输出的尺寸
    patch_size = 14  # 根据嵌入层卷积核大小
    expected_patches = (input_height // patch_size) * (input_width // patch_size)
    print(f"Expected patches: {expected_patches}")

    # 提取图像patch tokens
    image_features = feature_maps
    print("Image features shape:", image_features.shape)

    # 计算特征图的高度和宽度
    grid_height = input_height // patch_size
    grid_width = input_width // patch_size
    print(f"Grid size: {grid_height}x{grid_width}")

    # 使用所有特征通道的平均值
    all_features = image_features[0, :, :]
    all_features_2d = all_features.reshape(grid_height, grid_width, -1)
    all_features_np = all_features_2d.cpu().numpy()

    # 归一化所有通道
    for i in range(all_features_np.shape[2]):
        channel = all_features_np[:, :, i]
        channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        all_features_np[:, :, i] = channel

    # 上采样所有通道的特征图
    upsampled_all_features = np.zeros((original_height, original_width, all_features_np.shape[2]))
    for i in range(all_features_np.shape[2]):
        upsampled_all_features[:, :, i] = cv2.resize(
            all_features_np[:, :, i],
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR
        )

    # 计算所有通道的平均值
    mean_feature = np.mean(upsampled_all_features, axis=2)
    # mean_feature = (mean_feature - mean_feature.min()) / (mean_feature.max() - mean_feature.min() + 1e-8)
    # 将原始图像转换为numpy数组
    original_np = np.array(original_image)

    # 创建叠加图像 (50%原始图像 + 50%特征图)
    plt.figure(figsize=(12, 10))
    plt.imshow(original_np, alpha=ratio)
    feat_radio = 1-ratio
    plt.imshow(mean_feature, cmap='jet', alpha=feat_radio)
    plt.colorbar(label='Feature Activation')
    plt.title('Mean Feature Map Overlay (All Channels, 50% Original + 50% Feature Map)')
    plt.axis('off')
    plt.savefig('groot_mean_feature_all_channels_upsampled.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path="../GR00T-N1.5-3B",
        tune_llm=False,  # backbone's LLM
        tune_visual=False,  # backbone's vision tower
        tune_projector=False,  # action head's projector
        tune_diffusion_model=False,  # action head's DiT
    ).cuda()

    eagle_processor = build_eagle_processor(DEFAULT_EAGLE_PATH)

    text_list = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image-1>pick the pear from the counter and place it in the plate<|im_end|>\n<|im_start|>assistant\n']

    frames = Image.open("/home/robot/pi/code/visual_net/000000039769.jpg")
    frames = np.array(frames)[np.newaxis, :, :, :]
    frames_tensor = torch.from_numpy(frames).to(torch.float32) / 255.0
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]
    transform = T.Resize((224,224), interpolation=T.InterpolationMode.BILINEAR, antialias=True)
    frames_tensor = transform(frames_tensor)
    image = (frames_tensor.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()[0]
    image = Image.fromarray(image)
    image_inputs = [
        image
    ]

    batch = {}
    eagle_inputs = eagle_processor(
        text=text_list, images=image_inputs, return_tensors="pt", padding=True
    )
    for k, v in eagle_inputs.items():
        k = "eagle_" + k
        batch[k] = v.to('cuda')

    eagle_prefix = "eagle_"
    eagle_input = {
        k.removeprefix(eagle_prefix): v
        for k, v in batch.items()
        if k.startswith(eagle_prefix)
    }
    del eagle_input["image_sizes"]

    # eagle_output = model.backbone.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
    vlm_model = model.backbone.eagle_model

    outputs, vit_embeds = extra_features(vlm_model, input_ids=eagle_input["input_ids"], pixel_values=eagle_input["pixel_values"], attention_mask=eagle_input["attention_mask"])
    hidden_states = outputs['hidden_states'][-1]

    input_ids = eagle_input["input_ids"]
    selected = input_ids == vlm_model.image_token_index

    vit_states = hidden_states[selected]

    original_image = image_inputs[0]
    original_width, original_height = original_image.size
    visualize_feature_map({'features': vit_states.unsqueeze(0).float()}, original_height, original_width, original_image, ratio=0.4)
    visualize_feature_map({'features': vit_embeds.float()}, original_height, original_width,
                          original_image, ratio=0.4)

if __name__ == '__main__':
    main()