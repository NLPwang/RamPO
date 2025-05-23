# 🦾 RamPO

🧑‍💻 This repository provides the source code and deployment process for RamPO: Retrieval-Augmented Monte Carlo Tree Search Preference Optimization for Multi-Hop Question Answering.

## 💫 Quick Start

The Python libraries required for the experiment are provided in the `requirements.txt` file. The experimental environment can be set up by running:

```bash
pip install -r requirements.txt
```

### 1.Vector database Construction

Configure the appropriate paths in `method/retrieve.py`. 

Then, run the following command in the root directory to build the vector database:

```bash
python mctot/retrieve.py
```

### 2.Run phase 1: preference pair generation

```bash
python run_mcts.py
```

### 3.Run phase 2: Reasoning DPO training

```bash
python dpo_training.py
```

After completing Phase 2, the final selected results will be recorded in the folder created by phase 1.

### 4.Run phase 3: Reasoning CoT inference

```bash
python eval_src/rag_eval.py
```

## 🚀 Citation

Under Review and Coming Soon ...
