from util import Util
class PromptWrapper:
    @staticmethod
    def get_prompt(task, x = "", y = ""):
        prompt = task.cot_prompt_wrap(x, y)
        return prompt
    
    @staticmethod
    def get_question_decompos_prompt():
        return Util.read_txt("mctot/prompts/decompose_answer_prompt_template.txt")
    
    @staticmethod
    def get_question_decompos_itr_prompt():
        return Util.read_txt("mctot/prompts/decompose_answer_itr_prompt_template.txt")
    
    @staticmethod
    def get_question_rephrase_prompt():
        return Util.read_txt("mctot/prompts/rephrase_prompt_template.txt")
    
    @staticmethod
    def get_subquestion_answer_prompt():
        return Util.read_txt("mctot/prompts/subquestion_answer_prompt_template.txt")
    
    @staticmethod
    def get_document_analyse_prompt():
        return Util.read_txt("mctot/prompts/document_analyse_prompt_template.txt")
    
    @staticmethod
    def get_conclude_answer_prompt():
        post_retrieval_prompts = Util.read_json("mctot/prompts/post_retrieval_prompt_template_fix_action.json")
        examples = f"Task: Answer the given question step-by-step\n\n"
        for i in range(len(post_retrieval_prompts)):
            step = 0
            for key in post_retrieval_prompts[i]:
                if key == 'QUERY':
                    examples += f'Question: {post_retrieval_prompts[i][key]}\n'
                elif key == 'REPHRASED_QUERY':
                    examples += f'Step {step} Rephrased Question: {post_retrieval_prompts[i][key]}\n'
                elif key == 'Subquestions_1':
                    examples += f'Step {step} Subquestions_1: {post_retrieval_prompts[i][key]}\n'
                elif key == 'Subquestions_2':
                    examples += f'Step {step} Subquestions_2: {post_retrieval_prompts[i][key]}\n'
                elif key == 'DOCUMENT':
                    examples += f'Step {step} Document: {post_retrieval_prompts[i][key]}\n'
                elif key == 'ANSWER':
                    examples += f'Step {step} So the final answer is: {post_retrieval_prompts[i][key]}'
                step += 1
            examples += '\n\n'
        return examples
    
    @staticmethod
    def get_naive_cot_prompt():
        return Util.read_txt("mctot/prompts/naive_cot.txt")
    
    