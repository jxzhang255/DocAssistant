import numpy as np
from PIL import Image
import os
import sys

def map_attention_to_image(attention_map, image_size, grid_rows=3, grid_cols=4):
    """
    将注意力图映射到完整图像的空间上。
    
    Args:
        attention_map (np.ndarray): 注意力图，形状为 (patches, patch_size, patch_size)。
        image_size (tuple): 图像尺寸 (宽度, 高度)。
        grid_rows (int): 网格的行数。
        grid_cols (int): 网格的列数。
        
    Returns:
        np.ndarray: 完整图像的注意力图，形状为 (height, width)。
    """
    patch_size = attention_map.shape[1]  # 假设每个patch是正方形
    width, height = image_size
    patch_w, patch_h = width // grid_cols, height // grid_rows
    
    # 初始化完整的注意力图
    full_attention = np.zeros((height, width))
    
    for idx in range(attention_map.shape[0]):
        row = idx // grid_cols
        col = idx % grid_cols
        patch_attention = attention_map[idx]
        # 放大到patch区域大小
        patch_attention_resized = Image.fromarray(np.uint8(patch_attention * 255), mode='L').resize((patch_w, patch_h), resample=Image.BICUBIC)
        patch_attention_resized = np.array(patch_attention_resized) / 255.0
        # 赋值到完整图像的对应位置
        full_attention[row*patch_h:(row+1)*patch_h, col*patch_w:(col+1)*patch_w] = patch_attention_resized
    
    # 归一化整个注意力图
    full_attention = normalize_attention(full_attention)
    
    return full_attention

sys.path.append(os.path.abspath(__file__).rsplit('/', 4)[0])
from datasets import load_from_disk
import matplotlib.pyplot as plt
import seaborn as sns
from internvl.train.dataset import dynamic_preprocess
attn = np.load('/home/jxzhang/paper_codes/InternVL/internvl_chat/eval/vqa/attn_map/docvqa_attnmap.npy')
print(attn.shape)
attention_map = attn[1][23]
attention_map = (attention_map * 255).astype(np.uint8)
attention_map = np.kron(attention_map, np.ones((14, 14)))
print('attention_map:',attention_map)

full_attention_map = map_attention_to_image(attention_map, image_size, grid_rows, grid_cols)
test = load_from_disk('/home/jxzhang/datasets/docvqa_cached_extractive_all_lowercase_True_msr_False_extraction_v3_enumeration')['val'].to_list()
image = os.path.join('/home/jxzhang/datasets/DUE_Benchmark/DocVQA/pngs',test[0]['image'][10:])
print('image:',image)

image = Image.open(image).convert('RGBA')
# images = dynamic_preprocess(image, min_num=1, max_num=12,
#                                         image_size=14, use_thumbnail=False)
# print(len(images))
# image = images[1]
# 确保 attention_map 与图像尺寸一致
image_size = image.size[0]  # 假设图像为正方形
if attention_map.shape[0] != image_size or attention_map.shape[1] != image_size:
    attention_map = Image.fromarray(attention_map)
    attention_map = attention_map.resize((image_size, image_size), resample=Image.BICUBIC)
    attention_map = np.array(attention_map)

    # 应用色图
    cmap = plt.get_cmap('jet')
    attention_colored = cmap(np.array(attention_map)/255.0)[:, :, :3]  # 只取RGB通道
    attention_colored = (attention_colored * 255).astype(np.uint8)
    attention_colored = Image.fromarray(attention_colored).convert("RGBA")

# 创建热力图
# plt.figure(figsize=(8, 8))
# plt.imshow(image)
# # plt.imshow(attention_map, cmap='jet', alpha=0.5)
# plt.title(f'Layer {20} Head {0} Cross-Attention')
# plt.axis('off')
overlay = Image.blend(image, attention_colored, alpha=0.5)
overlay.save('/home/jxzhang/paper_codes/InternVL/internvl_chat/eval/vqa/attn_map/1.png')
# plt.savefig('/home/jxzhang/paper_codes/InternVL/internvl_chat/eval/vqa/attn_map/1.png', bbox_inches='tight', pad_inches=0)
# plt.show()