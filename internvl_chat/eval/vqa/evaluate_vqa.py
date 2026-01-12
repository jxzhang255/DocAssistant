import argparse
import itertools
import json
import os
import random
import subprocess
import time
from functools import partial
from typing import Optional
import sys
sys.path.append(os.path.abspath(__file__).rsplit('/', 3)[0])
import torch
from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from textvqa_eval import TextVQAAccuracyEvaluator
from tqdm import tqdm
from transformers import AutoTokenizer
from PIL import ImageFile
from datasets import load_from_disk
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

ds_collections = {
    'vqav2_val': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_val.jsonl',
        'question': 'data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/vqav2/v2_mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vqav2_testdev': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_testdev.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'okvqa_val': {
        'train': 'data/okvqa/okvqa_train.jsonl',
        'test': 'data/okvqa/okvqa_val.jsonl',
        'question': 'data/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/okvqa/mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val_ocr': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val_llava.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_val': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_val.jsonl',
        'question': 'data/vizwiz/vizwiz_val_questions.json',
        'annotation': 'data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_test': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_test.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'docvqa_val': {
        'train': '/home/jxzhang/datasets/ureader_json/test/test_docvqa_new.json',
        'test': '/home/jxzhang/datasets/DUE_Benchmark/DUE_jsons/docvqa_test_question_types.json',
        'annotation': '/home/jxzhang/datasets/ureader_json/test/test_v1.0.json',
        'metric': 'anls',
        'max_new_tokens': 1000,
    },
    'docvqa_test': {
        'train': '/home/jxzhang/datasets/ureader_json/test/test_docvqa_new.json',
        'test': '/home/jxzhang/datasets/ureader_json/test/test_docvqa_new.json',
        'metric': None,
        'max_new_tokens': 1000,
    },
    'chartqa_test_human': {
        'train': 'data/chartqa/train_human.jsonl',
        'test': '/home/jxzhang/datasets/DUE_Benchmark/ChartQA/chartqa/test_human.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 1000,
    },
    'chartqa_test_augmented': {
        'train': '/home/jxzhang/datasets/DUE_Benchmark/ChartQA/chartqa/train_augmented.jsonl',
        'test': '/home/jxzhang/datasets/DUE_Benchmark/ChartQA/chartqa/test_augmented.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 1000,
    },
    'gqa_testdev': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/test_balanced.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'gqa_testdev_llava': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/llava_gqa_testdev_balanced_qwen_format.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'ocrvqa_val': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_val.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_test': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ai2diagram_test': {
        'train': 'data/ai2diagram/train.jsonl',
        'test': 'data/ai2diagram/test_vlmevalkit.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'infographicsvqa_val': {
        'train': '/home/jxzhang/datasets/InfographicVQA/test_infovqa.json',
        'test': '/home/jxzhang/datasets/DUE_Benchmark/DUE_jsons/infographicVQA_test_category.json',
        'annotation': '/home/jxzhang/datasets/InfographicVQA/test_infovqa.json',
        'metric': 'anls',
        'max_new_tokens': 1000,
    },
    'infographicsvqa_test': {
        'train': 'data/infographicsvqa/train.jsonl',
        'test': 'data/infographicsvqa/test.jsonl',
        'annotation': 'data/infographicsvqa/infographicsVQA_test_v1.0.json',
        'metric': None,
        'max_new_tokens': 100,
    }
}


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
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

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
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


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]

    return pixel_values, questions, annotations


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, few_shot, input_size=224, dynamic_image_size=True,
                 use_thumbnail=False, max_num=6):
        # self.test = json.load(open(test))
        # with open(test,'r') as f:
        #     self.test = eval(f.read())
        with open(test,'r') as f:
            self.test = f.readlines()

        # self.test = load_from_disk('/home/jxzhang/datasets/docvqa_cached_extractive_all_lowercase_True_msr_False_extraction_v3_enumeration')['val'].to_list()
        # self.test = self.test
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.few_shot = few_shot
        self.max_num = max_num
        if few_shot > 0:
            self.train = json.load(open(test))
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = eval(self.test[idx])
        # data = self.test[idx]
        # image, question, answer = data['image_local_name'], data[
        #     'question'], data['answers']
        # image, question, answer = data['image'], data[
        #     'question'], data['answer']

        image, question, answer = data['image']['image'], data[
            'question'], data['answer']

        few_shot_prompt = ''
        if self.few_shot > 0:
            few_shot_samples = random.sample(self.train, self.few_shot)
            for sample in few_shot_samples:
                few_shot_prompt += self.prompt.format(
                    sample['image'],
                    sample['question']) + f" {sample['answer']}"
        
        # image = Image.open(os.path.join('/home/jxzhang/datasets/InfographicVQA/infographicVQA_val_v1.0_images',image)).convert('RGB')
        image = Image.open(os.path.join('/home/jxzhang/datasets/DUE_Benchmark/DocVQA/pngs',image[10:])).convert('RGB')
        # image = Image.open(os.path.join('/home/jxzhang/datasets/DUE_Benchmark/ChartQA/test',image)).convert('RGB')
        # image = Image.open(os.path.join('/home/jxzhang/datasets/DUE_Benchmark/InfographicsVQA/pngs',image)).convert('RGB')
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        print('pixel_values:',pixel_values.shape)
        if len(self.prompt) != 0:
            # The example format of response:
                # | step | output |
                # | 1 | {"SARS": "10%", "MERS": "34%", "EBOLA": "50%+"} |
                # | 2 | EBOLA |
                # The answer is: EBOLA
                #"Follow the format of the example of response, answer the question:\n"
            # prompt = "Below is an instruction that describes a question answering task in the general document domain, paired with an image. The given question is relevant to the image. Generate an appropriate answer to the given question.\n"
            # example = """
            # ### Instruction:
            #     Given a image and a question in the following, what is the answer to the question? Please complete the task in three steps:
            #     1. In the first step, extract the relevant contexts related to the keywords in the question from the provided image content Store these in the variable "{evidence}". If there are multiple contexts, separate them using the "#" symbol.
            #     2. In the second step, predict the answer based on the {evidence} and store it in the variable "{answer}".
            #     Please organize the results in the following table:
            #     | step | output |
            #     | 1 | {evidence} |
            #     | 2 | {answer} |
            #     Finally, present the predicted answer in the format: "The answer is: {answer}"
            # """
            # instruction = prompt+example+"\n"+"###Question\n"+question
            example = """
                Example1:
                    Generate the corresponding rationale and the answer based on the image and the question:
                    
                    Question: What is the difference in value between Lamb and Corn?
                    Rationale: 
                        1. The values of relevant indicators in the question are identified: The value of Lamb is 103.7 and the value of Corn is 103.13,
                        2. Perform the calculation of the difference in value between Lamb and Corn: 103.7-103.13=0.57,
                        3. The final calculation result is obtained: 0.57.
                    Answer from rationale: 0.57

                Example2:
                    Generate the corresponding rationale and the answer based on the image and the question:

                    Question: What's the average of all the values in the green bars?
                    Rationale: 
                        1. Identify all infomation of the blue bar: {"Characteristic": "US, EU, China", "More": "29, 19, 17"},
                        2. Perform the calculation of the average of all the values in the green bars: 29 + 19 + 17 = 21.6,
                        3. The final calculation result is obtained: 21.6.
                    Answer from rationale: 21.6
                
                Example3:
                    Generate the corresponding rationale and the answer based on the image and the question:

                    question: What is the sum of all the blue bar?
                    Rationale of step by step: 
                        1. Identify all infomation of the blue bar: {"Characteristic": "Number of gamers in millions", "2012": 8.12, "2013": 9.04, "2014": 9.97},
                        2. Calculate the sum of all values: 8.12+9.04+9.97=27.13,
                        3. The final calculation result is obtained: 27.13.
                    Answer from rationale: 27.13
                    """
            # example = """
            #         Example1:
            #             Question: Which disease has the highest mortality rate?
            #             Generate the corresponding rationale and answer based on the image and the question:
                        
            #             Rationales: {"COVID-19": "1-2%", "SARS": "10%", "MERS": "34%", "EBOLA": "50%+"}
            #             Answer from rationale: EBOLA

            #         Example2:
            #             Question: Where the Parklands traditional 21 gun salute event is happening?
            #             Generate the corresponding rationale and answer based on the image and the question:

            #             Rationale: {"name": "AUSTRALIA DAY FESTIVAL", "location": "South Bank, Brisbane", "date": "Sunday 26 January", "time": "10AM-7:30PM", "entry": "FREE ENTRY"}
            #             Answer from rationale: South Bank, Brisbane

            #         Example3:
            #             Question: How many Likes of David are listed?
            #             Generate the corresponding rationale and answer based on the image and the question:

            #             Rationale: {"Personal Likes": ["Process automation", "Attention to detail", "Designing online automated self-service systems", "Organized people", "Speed & excellence", "Having a visible work product"]}
            #             Answer from rationale: 6
            #     """
            # instruction = example+"\n"+"Follow the format of the example above, Generate the corresponding rationale and answer based on the image and the question:\n"+question
            instruction = "Answer the question using a single word:\n"+question
            # instruction = "Answer the question step by step:\n"+question
            # instruction= question
        return {
            'question': instruction,
            'pixel_values': pixel_values,
            'annotation': answer
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def post_process(response):
    response = response.strip().split('.')[0].split(
        ',')[0].split('!')[0].lower()
    if 'is ' in response:
        response = response.split('is ')[1]
    if 'are ' in response:
        response = response.split('are ')[1]
    if 'a ' in response:
        response = response.split('a ')[1]
    if 'an ' in response:
        response = response.split('an ')[1]
    if 'the ' in response:
        response = response.split('the ')[1]
    if ' of' in response:
        response = response.split(' of')[0]
    response = response.strip()
    return response


def evaluate_chat_model():
    base_prompt = 'Answer the question using a single word or phrase.'
    vizwiz_prompt = "When the provided information is insufficient, respond with 'Unanswerable'. "
    # infovqa_prompt = 'Answer the question directly.'
    infovqa_prompt = 'Answer the question using a single word or phrase.'
    ai2d_prompt = ''
    random.seed(args.seed)
    summaries = []

    for ds_name in args.datasets:
        if 'vizwiz' in ds_name:
            input_prompt = vizwiz_prompt + base_prompt
        elif 'ai2d' in ds_name:
            input_prompt = ai2d_prompt
        elif 'infographicsvqa' in ds_name:
            input_prompt = infovqa_prompt
        else:
            input_prompt = base_prompt

        dataset = VQADataset(
            train=ds_collections[ds_name]['train'],
            test=ds_collections[ds_name]['test'],
            prompt=input_prompt,
            few_shot=args.few_shot,
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        outputs = []
        for _, (pixel_values, questions, annotations) in tqdm(enumerate(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=1,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            pred = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=questions[0],
                generation_config=generation_config
            )
            answers = [pred]

            for question, answer, annotation in zip(questions, answers, annotations):
                if ds_name in ['vqav2_val', 'vqav2_testdev', 'okvqa_val', 'textvqa_val',
                               'vizwiz_val', 'textvqa_val_ocr']:
                    outputs.append({
                        'question': question,
                        'answer': answer,
                    })
                elif ds_name in ['docvqa_val', 'infographicsvqa_val', 'gqa_testdev', 'ocrvqa_val',
                                 'ocrvqa_test', 'gqa_testdev_llava', 'infographicsvqa_test',]:
                    print('answer:',annotation)
                    print("pred:",answer)
                    print()
                    outputs.append({
                        'question': question,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['ai2diagram_test']:
                    outputs.append({
                        'question': question,
                        'image': question_id,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['chartqa_test_human', 'chartqa_test_augmented']:
                    print('answer:',annotation)
                    print("pred:",answer)
                    print()
                    outputs.append({
                        'question': question,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['docvqa_test']:
                    outputs.append({
                        'question': question,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['vizwiz_test']:
                    outputs.append({
                        'image': question_id.replace('data/vizwiz/test/', ''),
                        'answer': answer,
                    })
                else:
                    raise NotImplementedError

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))
            print('Results saved to {}'.format(results_file))

            if ds_collections[ds_name]['metric'] == 'vqa_score':
                evaluator = TextVQAAccuracyEvaluator()
                annotation = json.load(open(ds_collections[ds_name]['annotation'], 'r'))
                question_id2answers = {}
                for item in annotation:
                    answers = item['answer']
                    # answers = [answer['answer'] for answer in item['answers']]
                    # question_id2answers[question_id] = answers
                for item in merged_outputs:
                    item['pred_answer'] = item['answer']
                    # item['gt_answers'] = question_id2answers[item['question_id']]s
                accuracy = evaluator.eval_pred_list(merged_outputs)
                print(ds_name, accuracy)
                summaries.append([args.checkpoint, ds_name, accuracy])

            elif ds_collections[ds_name]['metric'] == 'anls':
                json.dump(merged_outputs,
                          open(results_file, 'w'),
                          ensure_ascii=False)
                print('/home/jxzhang/paper_codes/InternVL/internvl_chat/eval/vqa/infographicsvqa_eval.py -g ' +
                      ds_collections[ds_name]['annotation'] + ' -s ' +
                      results_file)
                os.system('/home/jxzhang/paper_codes/InternVL/internvl_chat/eval/vqa/infographicsvqa_eval.py -g ' +
                          ds_collections[ds_name]['annotation'] + ' -s ' +
                          results_file)
            elif ds_collections[ds_name]['metric'] == 'relaxed_accuracy':
                relaxed_accuracy = evaluate_relaxed_accuracy(merged_outputs)
                print(ds_name, {'relaxed_accuracy': relaxed_accuracy})
                summaries.append([ds_name, {'relaxed_accuracy': relaxed_accuracy}])
            elif ds_collections[ds_name]['metric'] == 'accuracy':
                if 'gqa' in ds_name:
                    dst_file = './data/gqa/testdev_balanced_predictions.json'
                    print('python eval/vqa/convert_gqa_for_eval.py --src ' +
                          results_file + ' --dst ' + dst_file)
                    python_path = 'python'
                    os.system(python_path + ' eval/vqa/convert_gqa_for_eval.py --src ' +
                              results_file + ' --dst ' + dst_file)
                    command = f'cd ./data/gqa/ && {python_path} eval.py --tier testdev_balanced && cd ../../'
                    print(command)
                    accuracy = subprocess.check_output(command, shell=True, universal_newlines=True)
                else:
                    accuracy = {'accuracy': evaluate_exact_match_accuracy(merged_outputs)}
                print(ds_name, accuracy)
                summaries.append([args.checkpoint, ds_name, accuracy])

        torch.distributed.barrier()

    out_path = '_'.join(args.checkpoint.split('/')[-2:])
    writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    for summary in summaries:
        print(summary)
        writer.write(f'{summary}\n')
    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--datasets', type=str,)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str)
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true',default=True)
    parser.add_argument('--max-num', type=int, default=12)
    parser.add_argument('--load-in-8bit', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16).eval().cuda()
    # if not args.load_in_8bit:
    #     model = model.cuda()
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')

    evaluate_chat_model()
