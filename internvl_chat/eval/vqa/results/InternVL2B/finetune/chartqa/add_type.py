import json
from tqdm import tqdm
with open('/home/jxzhang/datasets/DUE_Benchmark/DUE_jsons/chart_rationales/human_test/Color.json','r') as f:
    color_data = f.readlines()
with open('/home/jxzhang/datasets/DUE_Benchmark/DUE_jsons/chart_rationales/human_test/Text_extractive.json','r') as f:
    text_data = f.readlines()

with open('/home/jxzhang/datasets/DUE_Benchmark/DUE_jsons/chart_rationales/human_test/Spatial.json','r') as f:
    spatial_data = f.readlines()

with open('/home/jxzhang/datasets/DUE_Benchmark/DUE_jsons/chart_rationales/human_test/Count.json','r') as f:
    count_data = f.readlines()

with open('/home/jxzhang/datasets/DUE_Benchmark/DUE_jsons/chart_rationales/human_test/Reasoning.json','r') as f:
    reasoning_data = f.readlines()

with open('/home/jxzhang/datasets/DUE_Benchmark/ChartQA/chartqa/test_human.jsonl','r') as f:
    data = f.read()
    data = eval(data)
all_data = []
for item in tqdm(data):
    data_item = {}
    for color_item in color_data:
        color_item = eval(color_item)
        if item['image'] in color_item['image'] and item['question'] == color_item['question']:
            data_item['image'] = item['image']
            data_item['question'] = item['question']
            data_item['answer'] = item['answer']
            data_item['question_type'] = 'Color'
    for text_item in text_data:
        text_item = eval(text_item)
        if item['image'] in text_item['image'] and item['question'] == text_item['question']:
            data_item['image'] = item['image']
            data_item['question'] = item['question']
            data_item['answer'] = item['answer']
            data_item['question_type'] = 'Text_extractive'
    for spatial_item in spatial_data:
        spatial_item = eval(spatial_item)
        if item['image'] in spatial_item['image'] and item['question'] == spatial_item['question']:
            data_item['image'] = item['image']
            data_item['question'] = item['question']
            data_item['answer'] = item['answer']
            data_item['question_type'] = 'Spatial'

    for count_item in count_data:
        count_item = eval(count_item)
        if item['image'] in count_item['image'] and item['question'] == count_item['question']:
            data_item['image'] = item['image']
            data_item['question'] = item['question']
            data_item['answer'] = item['answer']
            data_item['question_type'] = 'Count'

    for reasoning_item in reasoning_data:
        reasoning_item = eval(reasoning_item)
        if item['image'] in reasoning_item['image'] and item['question'] == reasoning_item['question']:
            data_item['image'] = item['image']
            data_item['question'] = item['question']
            data_item['answer'] = item['answer']
            data_item['question_type'] = 'Reasoning'
    all_data.append(data_item)

with open('/home/jxzhang/datasets/DUE_Benchmark/DUE_jsons/chart_rationales/human_test/chartqa_human_types.json','w') as f:
    json.dump(all_data,f)