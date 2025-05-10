from copy import deepcopy
from functools import reduce
import operator
from mctot.methods.prompt_wrapper import PromptWrapper
from util import Util
from llama_models import GPT
from mctot.retrieve import Retriever
import re


class MctsState:
    def __init__(self, query,  gpt:GPT, retriever: Retriever = None,parent = None,step = 0, TYPE: str = "QUERY", gen = "",disable_rag = False, solution_trace = [], rag_k = 3, iter_decompose = False,args = None):
        self.query = query
        self.gen = gen
        self.step = step
        self.TYPE = TYPE
        self.disable_rag = disable_rag
        self.parent = parent
        self.gpt = gpt
        self.retriever = retriever
        self.solution_trace = solution_trace
        self.rag_k = rag_k
        self.iter_decompose = iter_decompose
        self.args = args

    def getPossibleActions(self):
        if self.TYPE == "QUERY":
            return ["Rephrased Question"]
        elif self.TYPE == "Rephrased Question":
            return [ "Subquestions_1"]
        elif self.TYPE == "Subquestions_1":
            return [ "Subquestions_2"]
        elif self.TYPE == "Subquestions_2":
            return ["DOCUMENT"]
        elif self.TYPE == "DOCUMENT":
            return ["ANSWER"]
        
    def takeAction(self, type):
        # print("Action: ",str(type))
        solution_trace = [s for s in self.solution_trace]
        solution_trace.append(self)
        if type == "Rephrased Question":
            gen = Action.do_action_question_rephrase(self.query.strip(), self.gpt)[0].strip()
            return MctsState(gen.replace("<Rephrased Question>","").replace("</Rephrased Question>",""), self.gpt, self.retriever, self,  self.step + 1, type, gen,self.disable_rag, solution_trace, self.rag_k,self.iter_decompose,args=self.args)
        if "Subquestions" in type:
            if not self.iter_decompose:
                gen = Action.do_action_question_decompose(self.query.strip(),self.gpt,solution_trace,type)[0].strip()
            else:
                gen = Action.do_action_question_decompose_itr(self.query.strip(),self.gpt,solution_trace,type)[0].strip()
            if gen[-1] != '>':
                gen += f"</{type}>"
        if type == "DOCUMENT":
            prompt = ""
            for state in solution_trace:
                key = state.TYPE
                if key == 'QUERY':
                    prompt += f'{state.query.strip()}\n'
                else:
                    match = re.search(r'>(.*?)<', state.gen, re.DOTALL)
                    try:
                        query = match.group(1).strip().lower()
                    except Exception:
                        print("re ERRO ", state.gen)
                        query = state.gen
                    prompt += f'{query}\n'
            gen = Action.retrieve_documents(prompt, self.rag_k,self.retriever,self.gpt,self.args.document_analyse,self.query,args=self.args)
        if type == "ANSWER":
            prompt = ""
            for state in solution_trace:
                key = state.TYPE
                if key == 'QUERY':
                    prompt += f'Question: {state.query.strip()}\n'
                elif key == 'DOCUMENT':
                    prompt += f'Step {state.step} DOCUMENT: {state.gen.strip()}\n'
                else:
                    # prompt += f'{state.gen.strip()}\n'
                    match = re.search(r'>(.*?)<', state.gen, re.DOTALL)
                    try:
                        gen = match.group(1).strip().lower()
                    except Exception:
                        print("re ERRO ", state.gen)
                        gen = state.gen
                    prompt += f"Step {state.step} {state.TYPE}: {gen}\n"
            prompt += f"Step {self.step + 1}"
            gen = Action.conclude_answer(prompt, self.gpt,self.args)[0].strip()
        return MctsState(self.query, self.gpt, self.retriever, self,  self.step + 1, type, gen,self.disable_rag, solution_trace, self.rag_k,self.iter_decompose,args=self.args)

    def isTerminal(self):
        return self.TYPE == "ANSWER"

    def getReward(self,GR):
        if self.TYPE  != "ANSWER":
            return 0
        else:
            if str(GR).lower() in self.gen:
                return 1
            else:
                return 0

    
class Action():    
    @staticmethod
    def do_action_question_decompose(query,gpt,solution_trace,type):  # A1
        input = PromptWrapper.get_question_decompos_prompt()
        input += f'\n\n<Question>\n{query}\n</Question>\n'
        if type == "Subquestions_2":
            for s in solution_trace:
                if s.TYPE == "Subquestions_1":
                    input += s.gen.strip() + '\n'
                    prompt = input + '<Subquestions_2>\n'
        else: 
            prompt =  input + '<Subquestions_1>\n'
        solutions, _ = gpt.generate(prompt, stop_token_list=["</Subquestions_1>","</Subquestions_2>","\n","."])
        solutions = [s.replace(input, '') for s in solutions]
        return solutions
    
    @staticmethod
    def do_action_question_decompose_itr(query,gpt,solution_trace,type):  # A1
        input = PromptWrapper.get_question_decompos_itr_prompt()
        input += f'\n\n<Question>\n{query}\n</Question>\n'
        for s in solution_trace:
            if s.TYPE == "Subquestions_ITER" or s.TYPE == "DOCUMENT":
                input += s.gen.strip() + '\n'
        prompt = input + '<Subquestions>\n'
        
        solutions, _ = gpt.generate(prompt, stop_token_list=["</Subquestions>"])
        solutions = [s.replace(input, '') for s in solutions]
        return solutions
    
    @staticmethod
    def do_action_question_rephrase(query, gpt: GPT): # A2
        prefix = PromptWrapper.get_question_rephrase_prompt()
        prefix += f'\n\n<Original Question>\n{query}\n</Original Question>\n\n'
        prompt = prefix + f'<Rephrased Question>\n'
        solutions, _ = gpt.generate(prompt,stop_token_list=["</Rephrased Question>"])
        solutions = [s.replace(prefix, '') for s in solutions]
        return solutions
    

    def retrieve_documents(query, k, retriever: Retriever, gpt = None,document_analyse = False,q = None,args = None):  # A4

        _, indices = retriever.retrieve(query=[query], k= k)
        indices = indices[0][::-1]
        document = ''
        for i, idx in enumerate(indices):
            doc = retriever.get_document(index=int(idx))
            document += f' <doc>{doc}</doc>'
        if document_analyse:
            prompt = PromptWrapper.get_document_analyse_prompt()
            prompt += f'\n\nQuestion: {q}Document:{document}\nSummary: 1. '
            stop_token_list = ['\n']
            if 'qwen' in args.base_model.lower():
                stop_token_list=["\n\n"]
            solutions, _ = gpt.generate(prompt,stop_token_list = stop_token_list,do_sample = False)
            document = solutions[0].split("Summary:")[-1].strip()
        return document

    def conclude_answer(prompt, gpt: GPT,args):
        examples = PromptWrapper.get_conclude_answer_prompt()
        input = examples + prompt
        input += " So the final answer is: "
        stop_token_list=["\n"]
        if 'qwen' in args.base_model.lower():
            stop_token_list=[".","\n"]
        solutions, _ = gpt.generate(input, stop_token_list=stop_token_list)
        solutions = [s.split('So the final answer is: ')[-1].strip().lower() for s in solutions]
        return solutions