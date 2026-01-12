import json
import re
import numpy as np
def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def evaluate_anls(answer, gt_answers, threshold=0.5):
        '''DOcVQA, InfographicsVQA, STVQA'''
        answer = ' '.join(answer.strip().lower().split())
        if not isinstance(gt_answers, list):
            gt_answers = [gt_answers]
        gt_answers = [' '.join(gt_answer.strip().lower().split()) for gt_answer in gt_answers]

        values = []
        for gt_answer in gt_answers:
            dist = levenshtein_distance(answer, gt_answer)
            length = max(len(answer), len(gt_answer))
            values.append(0.0 if length == 0 else float(dist) / float(length))

        score = 1 - min(values)
        
        score = 0 if score < threshold else score
        
        return score
scores = []
def cal_anls_score(pred,answer):
    max_score = []
    for ans in answer:
        ans = ans.lower().strip()
        max_score.append(evaluate_anls(pred.lower().strip(),ans.lower().strip()))
    score = max(max_score)
    return score

    # flag = False
    # for ans in answer:
    #     ans = ans.lower().strip()
    #     if pred in ans or ans in pred:
    #         flag = True
    #         return 1
    # if flag == False:
    #     return 0

with open('/home/jxzhang/paper_codes/InternVL/internvl_chat/eval/vqa/results/InternVL2B/pretrain/infovqa/infographicsvqa_val_241111085301.json','r') as f:
    data = json.load(f)
for item in data:
    answer = item['annotation']
    pred = item['answer'].lower().strip()
    score = cal_anls_score(pred,answer)
    scores.append(score)
print(np.mean(scores))
# with open('/home/jxzhang/datasets/DUE_Benchmark/DUE_jsons/docvqa_test_question_types.json','r') as f:
#     type_data = f.readlines()
# with open('/home/jxzhang/paper_codes/InternVL/internvl_chat/eval/vqa/results/InternVL2B/finetune/infovqa/infographicsvqa_val_240822101954.json','r') as f:
#     data = json.load(f)
# scores = []
# text = 0
# count = 0
# spatial = 0
# reasoning = 0
# color = 0
# text_score = []
# count_score = []
# spatial_score = []
# reasoning_score = []
# color_score = []
# for item in data:
#     answer = item['annotation']
#     pred = item['answer'].lower().strip()
#     flag = False
#     for ans in answer:
#         ans = ans.lower().strip()
        
#         if ans in pred:
#             flag = True
#             scores.append(1)
#             # return score
            
#     if flag == False:
#         scores.append(0)
#     # scores.append(cal_anls_score(pred,answer,flag))
# print(np.mean(scores))
# for type, item in zip(type_data,data):
#     type = eval(type)
#     type = type['question_type']
#     answer = item['annotation']
#     pred = item['answer'].lower().strip()
#     if 'Text_extractive' in type:
#         text += 1
#         flag = False
#         for ans in answer:
#             ans = ans.lower().strip()
            
#             if ans in pred:
#                 flag = True
#                 text_score.append(1)
#                 # return score
                
#         if flag == False:
#             text_score.append(0)
#     if 'Count' in type:
#         count += 1
#         flag = False
#         for ans in answer:
#             ans = ans.lower().strip()
            
#             if ans in pred:
#                 flag = True
#                 count_score.append(1)
#                 # return score
                
#         if flag == False:
#             count_score.append(0)
#     if 'Spatial' in type:
#         spatial += 1
#         flag = False
#         for ans in answer:
#             ans = ans.lower().strip()
            
#             if ans in pred:
#                 flag = True
#                 spatial_score.append(1)
#                 # return score
                
#         if flag == False:
#             spatial_score.append(0)
#     if 'Reasoning' in type:
#         reasoning += 1
#         flag = False
#         for ans in answer:
#             ans = ans.lower().strip()
            
#             if ans in pred:
#                 flag = True
#                 reasoning_score.append(1)
#                 # return score
                
#         if flag == False:
#             reasoning_score.append(0)
#     if 'Color' in type:
#         color += 1
#         flag = False
#         for ans in answer:
#             ans = ans.lower().strip()
            
#             if ans in pred:
#                 flag = True
#                 color_score.append(1)
#                 # return score
                
#         if flag == False:
#             color_score.append(0)
# print(color,text,spatial,count,reasoning)


# print('color:',np.mean(color_score))
# print('text:',np.mean(text_score))
# print('spatial:',np.mean(spatial_score))
# print('count:',np.mean(count_score))
# print('reasoning:',np.mean(reasoning_score))       
    

    # else:
    #     print('pred:',pred)
    #     print('answer:',answer)
    #     print()
    # if "Answer from rationale" in pred:
    # # if 'Answer from rationale' in pred:
    #     match = re.search(r'([^:]*$)', pred)
    #     pred = match.group(1)
    #     for ans in answer:
    #         max_score.append(evaluate_anls(pred.lower().strip(),ans.lower().strip()))
    #     scores.append(max(max_score))
    # else:
    #     scores.append(0)
  

    # scores.append(evaluate_anls(pred,answer))
    # if answer in pred:
    #     pred = answer
    #     scores.append(1)
    # # elif "the answer is" in pred:
    # #     match = re.search(r'([^:]*$)', pred)
    # #     pred = match.group(1).strip()
    # #     print('pred:',pred)
    # #     scores.append(evaluate_anls(pred,answer))
    # else:
    #     scores.append(0)




# from typing import Optional
# import re
# def relaxed_correctness(target: str,
#                         prediction: str,
#                         max_relative_change: float = 0.05) -> bool:
#     """Calculates relaxed correctness.

#     The correctness tolerates certain error ratio defined by max_relative_change.
#     See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
#     “Following Methani et al. (2020), we use a relaxed accuracy measure for the
#     numeric answers to allow a minor inaccuracy that may result from the automatic
#     data extraction process. We consider an answer to be correct if it is within
#     5% of the gold answer. For non-numeric answers, we still need an exact match
#     to consider an answer to be correct.”

#     Args:
#       target: Target string.
#       prediction: Predicted string.
#       max_relative_change: Maximum relative change.

#     Returns:
#       Whether the prediction was correct given the specified tolerance.
#     """
#     # prediction = re.sub(r'[^a-zA-Z0-9. ]', '', prediction)
#     # target = re.sub(r'[^a-zA-Z0-9. ]', '', target)
#     # print(prediction)
#     # print(target)
#     def _to_float(text: str) -> Optional[float]:
#         try:
#             if text.endswith('%'):
#                 # Convert percentages to floats.
#                 return float(text.rstrip('%')) / 100.0
#             else:
#                 return float(text)
#         except ValueError:
#             print('error')
#             return None

#     prediction_float = _to_float(prediction)
#     # print('pred_float:',prediction_float)
#     target_float = _to_float(target)
#     if prediction_float is not None and target_float:
#         relative_change = abs(prediction_float -
#                               target_float) / abs(target_float)
#         return relative_change <= max_relative_change
#     else:
#         if "and" in pred:
#             prediction =target.lower()
#         return prediction.lower() == target.lower()


# def evaluate_relaxed_accuracy(entries):
#     scores = []
#     for elem in entries:
#         if isinstance(elem['annotation'], str):
#             elem['annotation'] = [elem['annotation']]
#         score = max([
#             relaxed_correctness(elem['answer'].strip(), ann)
#             for ann in elem['annotation']
#         ])
#         scores.append(score)
#     return sum(scores) / len(scores)

# with open('/home/jxzhang/paper_codes/InternVL/internvl_chat/eval/vqa/results/InternVL2B/chartqa_test_human_240711172051.json','r') as f:
#     data = json.load(f)
# with open('/home/jxzhang/datasets/DUE_Benchmark/DUE_jsons/chart_rationales/human_test/chartqa_human_types.json','r') as f:
#     type_data = json.load(f)
# scores = []
# for item in data:
#     answer = item['annotation'].lower().strip()
#     pred = item['answer'].lower().strip()
#     pred = re.sub(r'[^a-zA-Z0-9.: ]', '', pred)
#     answer = re.sub(r'[^a-zA-Z0-9.: ]', '', answer)
    

#     # if "the answer is" in pred:
#     #     match = re.search(r'([^:]*$)', pred)
#     #     pred = match.group(1).strip()
#     #     pred = re.sub(r'[^a-zA-Z0-9. ]', '', pred)
#     #     answer = re.sub(r'[^a-zA-Z0-9. ]', '', answer)
#     #     print('pred:',pred)
#     #     print('answer:',answer)
#     #     print()
#     #     if re.search(r'\d', pred):
#     #         # 只保留数字和小数点
#     #         pred = re.sub(r'[^0-9.]', '', pred)
#     #         pred = re.sub(r'\.([^0-9]|$)', '', pred)
#     #     print('pred:',pred)
#     #     print('answer:',answer)
#     #     # print()
#     #     if answer in pred or pred in answer:
#     #         print('score:',1)
#     #         scores.append(1)
#     #     else:
#     #         if isinstance(answer, str):
#     #             answer = [answer]
#     #         score = max([
#     #             relaxed_correctness(ann, pred)
#     #             for ann in answer
#     #         ])
#     #         print('score:',score)
#     # #     print(score)
#     # #     print()
#     #         scores.append(score)


#     if answer in pred:
#         scores.append(1)
#     else:
#         scores.append(0)
#         # if "the answer is" in pred:
#         #     match = re.search(r'([^:]*$)', pred)
#         #     pred = match.group(1).strip()
#         #     pred = re.sub(r'[^a-zA-Z0-9. ]', '', pred)
#         #     answer = re.sub(r'[^a-zA-Z0-9. ]', '', answer)
#         #     if isinstance(answer, str):
#         #         answer = [answer]
#         #     score = max([
#         #         relaxed_correctness(ann, pred)
#         #         for ann in answer
#         #     ])
#         #     scores.append(score)
#         # else:
#         #     scores.append(0)
#     # else:
    
#     #     if answer in pred:
#     #         scores.append(1)
#     #     else:
#     #         scores.append(0)
# print(np.mean(scores))


# text = 0
# count = 0
# spatial = 0
# reasoning = 0
# color = 0
# text_score = []
# count_score = []
# spatial_score = []
# reasoning_score = []
# color_score = []

# for type, item in zip(type_data,data):
#     # type = eval(type)
#     type = type['question_type']
#     answer = item['annotation'].lower().strip()
#     pred = item['answer'].lower().strip()
#     pred = re.sub(r'[^a-zA-Z0-9.: ]', '', pred)
#     answer = re.sub(r'[^a-zA-Z0-9.: ]', '', answer)
#     if 'Text_extractive' in type:
#         text += 1
#         if answer in pred:
#             text_score.append(1)
                
#         else:
#             text_score.append(0)
#     if 'Count' in type:
#         count += 1
#         if answer in pred:
#             count_score.append(1)
                
#         else:
#             count_score.append(0)
#     if 'Spatial' in type:
#         spatial += 1
#         if answer in pred:
#             spatial_score.append(1)
                
#         else:
#             spatial_score.append(0)
#     if 'Reasoning' in type:
#         reasoning += 1
#         if answer in pred:
#             reasoning_score.append(1)
                
#         else:
#             reasoning_score.append(0)
#     if 'Color' in type:
#         color += 1
#         if answer in pred:
#             color_score.append(1)
                
#         else:
#             color_score.append(0)
# print(color,text,spatial,count,reasoning)


# print('color:',np.mean(color_score))
# print('text:',np.mean(text_score))
# print('spatial:',np.mean(spatial_score))
# print('count:',np.mean(count_score))
# print('reasoning:',np.mean(reasoning_score)) 