from __future__ import division
from llama_models import GPT
import time
import math
import random
from methods.mcts_state import MctsState 
from collections import deque, Counter
import itertools
from util import Util
from functools import reduce
from mctot.methods.prompt_wrapper import PromptWrapper
import re

class treeNode():
    def __init__(self, state: MctsState, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}
        self.win_flag = 0
        self.valueVisits = 0
        
    def get_node_value(self):
        nodeValue = self.totalReward / self.valueVisits + 1 / math.sqrt(2) * math.sqrt(2 * math.log(self.parent.valueVisits) / self.valueVisits)
        return nodeValue


class MCTS():
    def __init__(self, gpt:GPT, GR, iterationLimit=None, explorationConstant=1 / math.sqrt(2), sample = 6, args = None,collect_strategy = ["winner-loser"]):
        self.explorationConstant = explorationConstant
        self.gpt = gpt
        self.GR = GR.lower()
        self.sample = sample
        self.answer_candidate = []
        self.iterationLimit = iterationLimit
        self.terminal_num = 0
        self.args = args
        self.collect_strategy = collect_strategy

    def search(self, initialState):
        self.root = treeNode(initialState, None)

        iter = 0
        if self.args.task == "2wiki":
            max_count = 50
        else:
            max_count = 40
        while self.terminal_num < max_count and iter < self.iterationLimit:
            if iter == 800:
                self.explorationConstant = self.explorationConstant * self.explorationConstant
            self.executeRound()
            iter += 1
        win_flag = False
        print(self.answer_candidate)
        for a in self.answer_candidate:
            if self.GR in a:
                win_flag = True
        if not win_flag:
            print("Fail to find the correct answer")
            return
        for cs in self.collect_strategy:
            self.getPair(self.root,cs)

    def executeRound(self):
        node = self.selectNode(self.root)
        reward = self.rollout(node)
        if node.isTerminal and reward > 0:
            trace = node
            while trace:
                trace.win_flag = 1
                trace = trace.parent
        self.backpropogate(node, reward)

    def selectNode(self, node):
        # print("selectNode")
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        # print("find Terminal node")
        return node

    def expand(self, node: treeNode):
        action_set = node.state.getPossibleActions()        
        explored_actions = list(node.children.keys())

        unexplored = []
        for action in action_set:
            if action not in explored_actions and action == 'DOCUMENT':
                unexplored.extend([action])
            elif action not in explored_actions:
                unexplored.extend([action] * self.sample)
            elif action != 'DOCUMENT':
                remaining = self.sample - len(node.children[action])
                if remaining > 0:
                    unexplored.extend([action] * remaining)
        action = random.choice(unexplored)
        if action not in node.children:
            node.children[action] = []
        new_state = node.state.takeAction(action)
        # print(action)
        newNode = treeNode(new_state, node)
        if newNode.isTerminal:
            self.terminal_num += 1
        node.children[action].append(newNode)
        is_fully_expanded = True
        for action in action_set:
            if (action not in node.children) or (len(node.children[action]) < self.sample and action != "DOCUMENT"):
                is_fully_expanded = False
                break
        node.isFullyExpanded = is_fully_expanded
        return newNode

    def backpropogate(self, node: treeNode, reward):
        # print("backpropogate")
        while node is not None:
            node.numVisits += 1
            node.valueVisits += 1 if reward >= 0 else 0
            node.totalReward += reward if reward >= 0 else 0
            node = node.parent

    def getBestChild(self, node, explorationValue):
        # print("getBestChild")
        bestValue = float("-inf")
        bestNodes = []
        for action, children_list in node.children.items():
            for child in children_list:
                # 计算UCT值
                nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(
                    2 * math.log(node.numVisits) / child.numVisits)
                
                if nodeValue > bestValue:
                    bestValue = nodeValue
                    bestNodes = [child]
                elif nodeValue == bestValue:
                    bestNodes.append(child)
        return random.choice(bestNodes)

    def getAction(self, root, bestChild):
        # print("getAction")
        for action, node in root.children.items():
            if node is bestChild:
                return action
            
    
    def rollout(self,node: treeNode):
        # print("rollout")
        if (node.numVisits > 1 and node.isTerminal): return -1
        current_state = node.state
        if current_state.isTerminal() :
            answer = current_state.gen
            self.answer_candidate.append(answer)
        while not current_state.isTerminal():
            action_set = current_state.getPossibleActions()
            action = random.choice(action_set)
            current_state = current_state.takeAction(action)
        reward = current_state.getReward(self.GR)
        return reward
    
    def adopt_self_consistence(self):
        count = Counter(self.answer_candidate)
        max_count = max(count.values())
        most_frequent = [element for element, cnt in count.items() if cnt == max_count]
        queue = deque()
        queue.append(self.root)
        while len(queue) > 0:
            node = queue.popleft()
            for action in node.children.values():
                for c in action:
                    queue.append(c)
            if node.isTerminal:
                Pre = node.state.gen
                if Pre in most_frequent:
                    current_node = node
                    while current_node:
                        current_node.win_flag = 1
                        current_node = current_node.parent


    def getPair(self, root:treeNode,type):
        
        def find_max_value_node(win_nodes):
            if not win_nodes:
                return None
            max_node = win_nodes[0]
            max_value = max_node.get_node_value()
            for node in win_nodes[1:]:
                current_value = node.get_node_value()
                if current_value > max_value:
                    max_node = node
                    max_value = current_value
            return max_node
        
        
        def find_min_value_node(lose_nodes):
            if not lose_nodes:
                return None
            min_node = lose_nodes[0]
            min_value = min_node.get_node_value()
            for node in lose_nodes[1:]:
                current_value = node.get_node_value()
                if current_value < min_value:
                    min_node = node
                    min_value = current_value
            return min_node
        
        step = 0
        queue = deque()
        queue.append(root)
        result_list = []
        while len(queue) > 0:
            node_candidate = []
            while len(queue) > 0 and queue[0].state.step == step:
                current_node = queue.popleft()
                for action in current_node.children.values():
                    for c in action:
                        queue.append(c)
                node_candidate.append(current_node)
            
            if len(node_candidate) > 1:
                win_nodes = []
                other_nodes = []
                for node in node_candidate:
                    if node.win_flag == 1:
                        win_nodes.append(node)
                    else:
                        other_nodes.append(node)
                        
                if len(win_nodes) and len(other_nodes):
                    if type == 'winner-loser':
                        for win, lose in list(itertools.product(win_nodes, other_nodes)):
                            if win.parent == lose.parent:
                                parent = win.parent
                                result = {
                                        "prompt": self.get_prompt(parent.state),
                                        "chosen": self.get_prompt(win.state).replace(self.get_prompt(parent.state),""),
                                        "rejected": self.get_prompt(lose.state).replace(self.get_prompt(parent.state),"")
                                    }
                                result_list.append(result)
                    elif type == 'highest-lowest':
                        
                        def find_max_min_pairs(win_nodes, other_nodes):
                            parent_map = {}
                            for node in win_nodes + other_nodes:
                                if node.parent not in parent_map:
                                    parent_map[node.parent] = {'win': [], 'other': []}
                                if node in win_nodes:
                                    parent_map[node.parent]['win'].append(node)
                                else:
                                    parent_map[node.parent]['other'].append(node)

                            pairs = []
                            for parent, nodes in parent_map.items():
                                win_nodes = nodes['win']
                                other_nodes = nodes['other']

                                if not win_nodes or not other_nodes:
                                    continue

                                max_node = find_max_value_node(win_nodes)
                                min_node = find_min_value_node(other_nodes)

                                if max_node and min_node:
                                    pairs.append((max_node, min_node))

                            return pairs
                
                        pairs = find_max_min_pairs(win_nodes,other_nodes)
                        for p in pairs:
                            win,lose = p
                            parent = win.parent
                            result = {
                                    "prompt": self.get_prompt(parent.state),
                                    "chosen": self.get_prompt(win.state).replace(self.get_prompt(parent.state),""),
                                    "rejected": self.get_prompt(lose.state).replace(self.get_prompt(parent.state),"")
                                }
                            result_list.append(result)
                    
                    elif type == 'highest-loser':
    
                        def find_max_lower_pairs(win_nodes, other_nodes):
                            parent_map = {}
                            for node in win_nodes + other_nodes:
                                if node.parent not in parent_map:
                                    parent_map[node.parent] = {'win': [], 'other': []}
                                if node in win_nodes:
                                    parent_map[node.parent]['win'].append(node)
                                else:
                                    parent_map[node.parent]['other'].append(node)

                            pairs = []
                            for parent, nodes in parent_map.items():
                                win_nodes = nodes['win']
                                other_nodes = nodes['other']

                                if not win_nodes or not other_nodes:
                                    continue

                                max_node = find_max_value_node(win_nodes)

                                if max_node:
                                    for l in other_nodes:
                                        pairs.append((max_node, l))

                            return pairs
                
                        pairs = find_max_lower_pairs(win_nodes,other_nodes)
                        for p in pairs:
                            win,lose = p
                            parent = win.parent
                            result = {
                                    "prompt": self.get_prompt(parent.state),
                                    "chosen": self.get_prompt(win.state).replace(self.get_prompt(parent.state),""),
                                    "rejected": self.get_prompt(lose.state).replace(self.get_prompt(parent.state),"")
                                }
                            result_list.append(result)
                            
                    elif type == 'winner-lowest':
    
                        def find_higher_min_pairs(win_nodes, other_nodes):
                            parent_map = {}
                            for node in win_nodes + other_nodes:
                                if node.parent not in parent_map:
                                    parent_map[node.parent] = {'win': [], 'other': []}
                                if node in win_nodes:
                                    parent_map[node.parent]['win'].append(node)
                                else:
                                    parent_map[node.parent]['other'].append(node)

                            pairs = []
                            for parent, nodes in parent_map.items():
                                win_nodes = nodes['win']
                                other_nodes = nodes['other']

                                if not win_nodes or not other_nodes:
                                    continue

                                min_node = find_min_value_node(win_nodes)

                                if min_node:
                                    for w in win_nodes:
                                        pairs.append((w, min_node))

                            return pairs
                
                        pairs = find_higher_min_pairs(win_nodes,other_nodes)
                        for p in pairs:
                            win,lose = p
                            parent = win.parent
                            result = {
                                    "prompt": self.get_prompt(parent.state),
                                    "chosen": self.get_prompt(win.state).replace(self.get_prompt(parent.state),""),
                                    "rejected": self.get_prompt(lose.state).replace(self.get_prompt(parent.state),"")
                                }
                            result_list.append(result)
                        
            step += 1
        run_function = lambda x, y: x if y in x else x + [y]
        new_result_list = reduce(run_function, [[], ] + result_list)
        print(f"{type} Generate {len(new_result_list)} samples")
        for result in new_result_list:
            Util.append_json_to_file(result, self.args.output_file.replace('.json',f'_{type}.json'))
        

            
            
    def get_prompt(self,state:MctsState):
        prompt = PromptWrapper.get_conclude_answer_prompt()
        for s in state.solution_trace:
            key = s.TYPE
            if key == 'QUERY':
                    prompt += f'Question: {s.query.strip()}\n'
            elif key == 'DOCUMENT':
                prompt += f'Step {s.step} DOCUMENT: {s.gen.strip()}\n'
            elif key == "ANSWER":
                prompt += f'Step {s.step} So the final answer is: {s.gen.strip()}\n'
            else:
                # prompt += f'{state.gen.strip()}\n'
                match = re.search(r'>(.*?)<', s.gen, re.DOTALL)
                try:
                    gen = match.group(1).strip().lower()
                except Exception:
                    print("re ERRO ", s.gen)
                    gen = s.gen
                prompt += f"Step {s.step} {s.TYPE}: {gen}\n"
        key = state.TYPE
        if key == 'QUERY':
                prompt += f'Question: {state.query.strip()}\n'
        elif key == 'DOCUMENT':
            prompt += f'Step {state.step} DOCUMENT: {state.gen.strip()}\n'
        elif key == "ANSWER":
            prompt += f'Step {state.step} So the final answer is: {state.gen.strip()}\n'
        else:
            # prompt += f'{state.gen.strip()}\n'
            match = re.search(r'>(.*?)<', state.gen, re.DOTALL)
            try:
                gen = match.group(1).strip().lower()
            except Exception:
                gen = state.gen
            prompt += f"Step {state.step} {state.TYPE}: {gen}\n"
        return prompt
