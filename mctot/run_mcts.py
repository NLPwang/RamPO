import argparse
from llama_models import GPT
import pandas as pd
from constant import *
import torch
from mctot.methods.mcts import MCTS
from mctot.methods.mcts_state import MctsState
from mctot.methods.prompt_wrapper import PromptWrapper
from tqdm import tqdm
import re
from mctot.retrieve import Retriever
from collections import Counter
import random
from evaluate_util import evaluate
from util import Util
def run(args):
    print('===base_model===')
    base_model = args.base_model
    print(base_model)
    
    if args.task == 'bamboogle':
        database_pth = "mctot/data/bamboogle.json"
        index_pth = "mctot/database/bamboogle.index"
    elif args.task == 'hotpotqa':
        database_pth = "mctot/data/hotpot_dev_distractor_v1.json"
        index_pth = "mctot/database/hotpot_dev_distractor_v1.index"       
    elif args.task == '2wiki':
        database_pth = "mctot/data/wiki_dev.json"
        index_pth = "mctot/database/wiki_dev.index" 
    retriever = Retriever(model_pth=args.retriever_pth, data_pth=database_pth)
    retriever.load_index(index_pth=index_pth)
    gpt = GPT(args)
    if args.task in ['bamboogle', '2wiki', 'hotpotqa']:
        if args.task == 'bamboogle':
            if not args.eval:
                data = list(pd.read_csv(bamboogle_DIR)['Question'])
                ground_truths = list(pd.read_csv(bamboogle_DIR)['Answer'])
            else:
                data = list(pd.read_csv(bamboogle_eval_DIR)['Question'])
                ground_truths = list(pd.read_csv(bamboogle_eval_DIR)['Answer'])
            data_dic = {}
            for i in range(len(data)):
                data_dic[data[i]] = ground_truths[i]
        if args.task == 'hotpotqa':
            if not args.eval:
                data = list(pd.read_csv(hotpotqa_DIR)['Question'])
                ground_truths = list(pd.read_csv(hotpotqa_DIR)['Answer'])
            else:
                data = list(pd.read_csv(hotpotqa_eval_DIR)['Question'])
                ground_truths = list(pd.read_csv(hotpotqa_eval_DIR)['Answer'])
            data_dic = {}
            for i in range(len(data)):
                data_dic[data[i]] = ground_truths[i]
            args.task = 'bamboogle'
        if args.task == '2wiki':
            if not args.eval:
                data = list(pd.read_csv(twowiki_DIR)['Question'])
                ground_truths = list(pd.read_csv(twowiki_DIR)['Answer'])
            else:
                data = list(pd.read_csv(twowiki_eval_DIR)['Question'])
                ground_truths = list(pd.read_csv(twowiki_eval_DIR)['Answer'])
            data_dic = {}
            for i in range(len(data)):
                data_dic[data[i]] = ground_truths[i]
            args.task = 'bamboogle'
                
    data = torch.utils.data.DataLoader(data,batch_size=1)
    data = gpt.accelerator.prepare_data_loader(data)
    
    if not args.eval:
        for d in tqdm(data,total=len(data)):
            for d_idx, d_i in enumerate(d):
                mcts = MCTS(gpt, data_dic[d_i], iterationLimit=1000,args=args,sample=6,collect_strategy=args.collect_strategy)
                initial_state = MctsState(d_i,gpt,retriever=retriever,iter_decompose=args.decompose_iter,args=args)
                mcts.search(initial_state)
    else:
        log = []
        for d in tqdm(data,total=len(data)):
            for d_idx, d_i in enumerate(d):
                answers = {}
                for i in range(args.sample):
                    examples = PromptWrapper.get_conclude_answer_prompt()
                    prompt = examples + f"Question: {d_i}\n"
                    solutions, _ = gpt.generate(prompt,stop_token_list=['Step 4'],do_sample = args.do_sample)
                    if not args.disable_rag:
                        query = extract_steps(solutions[0].replace(PromptWrapper.get_conclude_answer_prompt(),""))
                        _, indices = retriever.retrieve(query=[query], k= 3)
                        indices = indices[0][::-1]
                        document = ''
                        for i, idx in enumerate(indices):
                            doc = retriever.get_document(index=int(idx))
                            document += f' <doc>{doc}</doc>'
                        if args.document_analyse:
                            prompt = PromptWrapper.get_document_analyse_prompt()
                            prompt += f'\n\nQuestion: {d_i}Document:{document}\nSummary: 1. '
                            stop_token_list = ['\n']
                            if 'qwen' in args.base_model.lower():
                                stop_token_list=["\n\n"]
                            s, _ = gpt.generate(prompt,stop_token_list = stop_token_list,do_sample = False)
                            document = s[0].split("Summary:")[-1].strip()
                    else:
                            document = 'None'
                    prompt = solutions[0] + f" DOCUMENT: {document}\nStep 5 So the final answer is: "
                    stop_token_list=["\n"]
                    if 'qwen' in args.base_model.lower():
                        stop_token_list=[".","\n"]
                    solutions, _ = gpt.generate(prompt,stop_token_list=stop_token_list,do_sample = args.do_sample)
                    solution = solutions[0]
                    answer = solution.split("5 So the final answer is:")[-1].strip().replace('.',"")
                    answers[solution] = answer
                count = Counter(answers.values())
                max_count = max(count.values())
                most_frequent = [element for element, cnt in count.items() if cnt == max_count]
                answer = random.choice(most_frequent)
                solutions = [k for k, v in answers.items() if v == answer]
                print(solutions, '\n', f"====GR===={data_dic[d_i]}====Pre===={answer}")
                log.append(f"====GR===={data_dic[d_i]}====Pre===={answer}")
        evaluate(log)
        
                
def extract_steps(text):
    query = ""
    step = text.split('\n')
    for s in step[:4]:
        query += " " + s.replace('Question:',"").replace('Step 1 Rephrased',"").replace('Step 2 Subquestions_1:',"").replace('Step 3 Subquestions_2:',"").strip()
    return query

def extract_steps_list(text):
    query = []
    step = text.split('\n')
    for s in step[:4]:
        query.append(s.replace('Question:',"").replace('Step 1 Rephrased',"").replace('Step 2 Subquestions_1:',"").replace('Step 3 Subquestions_2:',"").strip())
    return query


                    
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--task', type=str, required=True, choices=['bamboogle', '2wiki', 'hotpotqa'])
    args.add_argument('--base_model', type=str, default='Llama3b')
    args.add_argument('--retriever_pth', type=str, default='your bge-m3 model')
    args.add_argument('--output_file', type=str)   
    args.add_argument('--eval', action='store_true')
    args.add_argument('--document_analyse', action='store_true')
    args.add_argument('--do_sample', type=bool, default=True)
    args.add_argument('--sample', type=int,default=5)
    args.add_argument('--collect_strategy', nargs='+', type=str, choices=['winner-loser','highest-lowest','highest-loser','winner-lowest'], default='winner-loser')
    args = args.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    print(args)
    run(args)