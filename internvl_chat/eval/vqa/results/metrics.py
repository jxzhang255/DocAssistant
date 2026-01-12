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

# with open('/home/jxzhang/paper_codes/InternVL/internvl_chat/eval/vqa/results/docvqa_val_240701160418.json','r') as f:
#     data = json.load(f)
# scores = []
# for item in data:
#     answer = item['annotation'].lower().strip()
#     pred = item['answer'].lower().strip()
#     max_score = []
#     # if "Answer from rationale" in pred:
#     # # if 'Answer from rationale' in pred:
#     #     match = re.search(r'([^:]*$)', pred)
#     #     pred = match.group(1)
#     #     for ans in answer:
#     #         max_score.append(evaluate_anls(pred.lower().strip(),ans.lower().strip()))
#     #     scores.append(max(max_score))
#     # else:
#     #     scores.append(0)
  

#     scores.append(evaluate_anls(pred,answer))
#     # if answer in pred:
#     #     pred = answer
#     #     scores.append(1)
#     # # elif "the answer is" in pred:
#     # #     match = re.search(r'([^:]*$)', pred)
#     # #     pred = match.group(1).strip()
#     # #     print('pred:',pred)
#     # #     scores.append(evaluate_anls(pred,answer))
#     # else:
#     #     scores.append(0)

# print(np.mean(scores))



from typing import Optional
import re
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined -by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """
    # prediction = re.sub(r'[^a-zA-Z0-9.]', '', prediction)
    # target = re.sub(r'[^a-zA-Z0-9.]', '', target)
    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            print('error')
            return None

    prediction_float = _to_float(prediction)
    print('pred_float:',prediction_float)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        if "and" in pred:
            prediction =target.lower()
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)

with open('/home/jxzhang/paper_codes/InternVL/internvl_chat/eval/vqa/results/InternVL2B/finetune/chartqa/chartqa_test_human_240822104122.json','r') as f:
    data = json.load(f)
scores = []
for item in data:
    answer = item['annotation'].lower().strip()
    pred = item['answer'].lower().strip()

    # if "the answer is:" in pred:
    #     match = re.search(r'([^:]*$)', pred)
    #     pred = match.group(1).strip()
    #     print('pred:',pred)
    #     print('answer:',answer)
        
    #     if isinstance(answer, str):
    #         answer = [answer]
    #     score = max([
    #         relaxed_correctness(ann, pred)
    #         for ann in answer
    #     ])
    #     print(score)
    #     print()
    #     scores.append(score)

    if answer in pred:
        scores.append(1)
    else:
        scores.append(0)
    # else:
    #     scores.append(0)
print(np.mean(scores))