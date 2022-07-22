"""
This file performs the following steps:
- Take a set of generated multiple-choice questions and corresponding contexts and prepare them in the form of a json
- Take the RACE++ dataset and setup in an equivalent json where the 1st option rearranged to be the correct one and only 1 question taken per context
- Confirm same number of questions in the generated and fake sets
"""

import argparse
import os
import sys
import json
from transformers import T5Tokenizer

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--gen_questions_path', type=str,  help='Specify path to generated questions on training set')
parser.add_argument('--gen_contexts_path', type=str,  help='Specify path to contexts corresponding to the generated questions')
parser.add_argument('--train_data_path', type=str, help='Load path of training data of RACE++')
parser.add_argument('--save_dir', type=str,  help='Specify path to save generated jsons')

# Everything will be tokenized and de-tokenized to make consistent
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def organise_data(questions, contexts):
    organised_data = []
    count = 0
    for question, context in zip(questions, contexts):
        count += 1
        print(count, len(questions))
        first_sep_pos = question.find("[SEP]")
        qu = question[:first_sep_pos]
        opts = []
        validSEP = True
        sep_pos = first_sep_pos
        while validSEP:
            question = question[sep_pos+6:]
            sep_pos = question.find("[SEP]")
            if sep_pos == -1:
                validSEP = False
                opt = question
            else:
                opt = question[:sep_pos]
            opts.append(tokenizer.decode(tokenizer.encode(opt), skip_special_tokens=True, clean_up_tokenization_spaces=True))
        curr_point = {'question': tokenizer.decode(tokenizer.encode(qu), skip_special_tokens=True, clean_up_tokenization_spaces=True), 'context': context, 'options':opts, 'label':0}
        # print(curr_point)
        organised_data.append(curr_point)
    return organised_data

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Let's prepare the real data first

    with open(args.train_data_path + "middle.json") as f:
        middle_data = json.load(f)
    with open(args.train_data_path + "high.json") as f:
        high_data = json.load(f)
    with open(args.train_data_path + "college.json") as f:
        college_data = json.load(f)
    train_data = middle_data + high_data + college_data

    def asNum(x):
        if x=="A":
            return 0
        if x=="B":
            return 1
        if x=="C":
            return 2
        if x=="D":
            return 3

    real_data = []

    to_remove_from_fake = []
    for train_count, item in enumerate(train_data):
        print(train_count, len(train_data))
        context = item["article"]
        # Take only the first question
        if not len(item["questions"]) > 0:
            print(item["questions"])
            to_remove_from_fake.append(train_count)
            continue
        question = item["questions"][0]
        answer = asNum(item["answers"][0])
        options = item["options"][0]
        # Rearrange to make the correct answer option first
        new_opts = [options[answer]]
        for opt_count, opt in enumerate(options):
            if opt_count == answer:
                continue
            new_opts.append(tokenizer.decode(tokenizer.encode(opt), skip_special_tokens=True, clean_up_tokenization_spaces=True))
        curr_item = {"question": tokenizer.decode(tokenizer.encode(question), skip_special_tokens=True, clean_up_tokenization_spaces=True), "context": context, "options": new_opts, "label":0}
        real_data.append(curr_item)

    # Now let's prepare the fake data

    with open(args.gen_questions_path, 'r') as f:
        all_gen_questions = [a.rstrip() for a in f.readlines()]
    with open(args.gen_contexts_path, 'r') as f:
        all_contexts = [a.rstrip() for a in f.readlines()]

    fake_data = organise_data(all_gen_questions, all_contexts)
    # Remove the examples missing from the real data
    to_remove_from_fake = to_remove_from_fake[::-1]
    for val in to_remove_from_fake:
        _ = fake_data.pop(val)

    # make sure formatting of contexts of real and fake match exactly
    real_data_corrected = []
    for real_item, fake_item in zip(real_data, fake_data):
        curr_item = real_item
        curr_item["context"] = fake_item["context"]
        real_data_corrected.append(curr_item)
    real_data = real_data_corrected


    # Check that there is the same amount of data in both files
    if len(fake_data) == len(real_data):
        print("Lengths match :)")
    else:
        print("Something went wrong :(")
        print(len(fake_data))
        print(len(real_data))

    with open(args.save_dir+'real.json', 'w') as f:
        json.dump(real_data, f)
        
    with open(args.save_dir+'fake.json', 'w') as f:
        json.dump(fake_data, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)