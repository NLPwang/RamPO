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
python mctot/run_mcts.py --task hotpotqa --output_file output/hotpotqa_preference_pairs.json --document_analyse --collect_strategy winner-loser
```

### 3.Run phase 2: Reasoning DPO training

```bash
python dpo_training.py --dataset output/hotpotqa_preference_pairs.json --wandb_name RamPO_hotpot_checkpoint  --output_dir checkpoints/RamPO_hotpot_checkpoint
```


### 4.Run phase 3: Reasoning CoT inference

```bash
python mctot/run_mcts.py --task hotpotqa --base_model checkpoints/RamPO_hotpot_checkpoint/checkpoint-7480  --document_analyse --eval > logs/hotpotqa_checkpoints/RamPO_hotpot_checkpoint_checkpoint-7480.out
```

## 🚀 Citation

Under Review and Coming Soon ...
