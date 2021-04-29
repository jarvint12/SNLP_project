

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model=AutoModelForQuestionAnswering.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model.load_state_dict(torch.load('bert_cased3_2_model_states.pth', map_location=torch.device(device)))#, map_location=lambda))


model.eval()

from transformers.data.processors.squad import SquadV2Processor
import os

# this processor loads the SQuAD2.0 dev set examples
processor = SquadV2Processor()
examples = processor.get_dev_examples("./", filename="dev-v2.0.json")
print(len(examples))

# generate some maps to help us identify examples of interest
qid_to_example_index = {example.qas_id: i for i, example in enumerate(examples)}
qid_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
answer_qids = [qas_id for qas_id, has_answer in qid_to_has_answer.items() if has_answer]
no_answer_qids = [qas_id for qas_id, has_answer in qid_to_has_answer.items() if not has_answer]

def get_prediction(qid):
    # given a question id (qas_id or qid), load the example, get the model outputs and generate an answer
    question = examples[qid_to_example_index[qid]].question_text
    context = examples[qid_to_example_index[qid]].context_text

    inputs = tokenizer.encode_plus(
                        question,context,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 512,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                        truncation=True
                   ).to(device)


    outputs = model(**inputs)
    answer_start = torch.argmax(outputs[0])  # get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(outputs[1]) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

    return answer


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)

def get_gold_answers(example):
    """helper function that retrieves all possible true answers from a squad2.0 example"""

    gold_answers = [answer["text"] for answer in example.answers if answer["text"]]

    # if gold_answers doesn't exist it's because this is a negative example -
    # the only correct answer is an empty string
    if not gold_answers:
        gold_answers = [""]

    return gold_answers

answer_qids[10]

import numpy as np
import torch
from tqdm import tqdm
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)


em_list = []
f1_list = []
save_pred = {}
for i in tqdm(range(len(answer_qids))):
    prediction = get_prediction(answer_qids[i])
    example = examples[qid_to_example_index[answer_qids[i]]]
    save_pred[answer_qids[i]] = prediction
    gold_answers = get_gold_answers(example)

    em_score = max((compute_exact_match(prediction, answer)) for answer in gold_answers)
    f1_score = max((compute_f1(prediction, answer)) for answer in gold_answers)

    if i<50:
        print(examples[qid_to_example_index[answer_qids[i]]].question_text, normalize_text(prediction), gold_answers, normalize_text(gold_answers[0]), em_score, f1_score)

    f1_list.append(f1_score)
    em_list.append(em_score)


f1 = np.mean(f1_list)
em = np.mean(em_list)

print(f1)
print(em)

import pickle
def save_obj(obj, name ):
    with open('squad/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('squad/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

save_obj(save_pred, "predicions_bert_cased_test2" )

len(examples[qid_to_example_index[answer_qids[1500]]].context_text.split())
question = examples[qid_to_example_index[answer_qids[i]]].question_text
context = examples[qid_to_example_index[answer_qids[i]]].context_text

inputs = tokenizer.encode_plus(question, context, return_tensors='pt').to(device)

len(inputs[0])
