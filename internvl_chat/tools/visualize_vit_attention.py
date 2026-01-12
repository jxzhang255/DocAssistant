import argparse
import os

import torch
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor

from internvl.model.internvl_chat import InternVLChatModel
from internvl.model.internvl_chat.modeling_intern_vit import (
    save_attention_heatmap,
    vit_last_layer_cls_attention_map,
)


def main():
    parser = argparse.ArgumentParser(description='Visualize last-layer ViT attention (CLS->patch) as heatmap')
    parser.add_argument('model_path', type=str, help='Path to InternVLChatModel checkpoint')
    parser.add_argument('image_path', type=str, help='Input image path')
    parser.add_argument('output_path', type=str, help='Output heatmap file path, e.g. out.png')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--overlay', action='store_true', help='Overlay heatmap on resized RGB image')
    parser.add_argument('--alpha', type=float, default=0.45, help='Overlay alpha')
    args = parser.parse_args()

    dtype = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }[args.dtype]

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    model = InternVLChatModel.from_pretrained(args.model_path, torch_dtype=dtype).eval()
    model = model.to(device)

    image_size = model.config.force_image_size or model.config.vision_config.image_size
    patch_size = model.config.vision_config.patch_size

    image_processor = CLIPImageProcessor(
        crop_size=image_size,
        do_center_crop=True,
        do_normalize=True,
        do_resize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        size=image_size,
    )

    pil = Image.open(args.image_path).convert('RGB')
    pixel_values = image_processor(images=pil, return_tensors='pt')['pixel_values']  # [1,3,H,W]
    pixel_values = pixel_values.to(device=device, dtype=dtype)

    with torch.no_grad():
        _, _, _, all_attn = model.vision_model(
            pixel_values=pixel_values,
            text_embeds=None,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )

    if all_attn is None or len(all_attn) == 0 or all_attn[-1] is None:
        raise RuntimeError(
            'No attentions returned. 请确认 vision_config.use_flash_attn=True 时已安装 flash-attn，'
            '并且我们代码路径能够在 flash-attn 下返回权重；或者临时把 use_flash_attn 设为 False。'
        )

    last_attn = all_attn[-1]
    attn_map = vit_last_layer_cls_attention_map(last_attn, pixel_values, patch_size=patch_size)

    overlay_img = None
    if args.overlay:
        # 使用未归一化的可视化图像（resize 到 image_size）
        vis_pil = pil.resize((image_size, image_size), resample=Image.BICUBIC)
        vis = torch.from_numpy(np.array(vis_pil).astype(np.float32) / 255.0)
        overlay_img = vis

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    save_attention_heatmap(attn_map, args.output_path, image=overlay_img, alpha=args.alpha)
    print(f'Saved: {args.output_path}')


if __name__ == '__main__':
    main()
