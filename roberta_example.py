import json
import torch
from fairseq.models.roberta import RobertaModel
from examples.roberta import commonsense_qa # load the Commonsense QA task

import sys
import random

use_random = len(sys.argv) > 1 and sys.argv[1] == "random"

if not use_random:
    roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint_best.pt', 'data/CommonsenseQA')
    roberta.eval() # disable dropout
    roberta.cuda() # using GPU
nsamples, ncorrect = 0, 0

with open('data/CommonsenseQA/valid.jsonl') as h: #valid.jsonl') as h:
    for line in h:
        example = json.loads(line)
        scores = []

        #print(example)

        if use_random:
            pred = random.randint(0, len(example['question']['choices']))
        else:
            for choice in example['question']['choices']:
                input = roberta.encode(
                    'Q: ' + example['question']['stem'],
                    'A: ' + choice['text'],
                    no_separator=True
                )
                score = roberta.predict('sentence_classification_head', input, return_logits=True)
                scores.append(score)
            pred = torch.cat(scores).argmax()

        answer = ord(example['answerKey']) - ord('A')
        nsamples += 1
        if pred == answer:
            ncorrect += 1

print(f'Accuracy: {ncorrect / float(nsamples)}')