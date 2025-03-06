# OSWorld Setup Guide

## Cloning the Repository

First, navigate to `llm-reasoners/examples/OSWorld` and clone the repository:

```bash
git clone https://github.com/anananan116/OSWorld.git
cd OSWorld
pip install -e # make sure to run this in llm-reasoners/examples/OSWorld/OSWorld 
```

## Installing Conda and Creating a Python Environment

Ensure you have Conda installed. 
Create and activate a Python 3.10 environment:

```bash
conda create -p /example/prefix/osworld python=3.10
conda activate osworld
```

- We recommend using prefixes for your conda environment creation, especially
on servers with limited disk space.

## Installing Dependencies

Navigate to the repository directory and install all required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Experiment

### Using GPT-4o

If you are using GPT-4o, export your OpenAI API key and run the following command:

```bash
cd 'llm-reasoners/examples/OSWorld/OSWorld'
export OPENAI_API_KEY=your_openai_api_key
python run.py --headless --observation_type screenshot --model gpt-4o --result_dir ./results
```

### Using UITARS (Recommended)

Start a UITARS inference vllm server:

```bash
cd 'llm-reasoners/examples/OSWorld/OSWorld'
python -m vllm.entrypoints.openai.api_server --served-model-name ui-tars --model /data/zihanliu/UI-TARS --limit-mm-per-prompt image=5 -tp 2 --port 8009
```

On another terminal tab, run:

```bash
cd 'llm-reasoners/examples/OSWorld/OSWorld'
python run_uitars.py --headless --observation_type screenshot --model ui-tars --result_dir ./results
```

### Using OSWorld + LLM-Reasoners for Search & Scaling
```bash
cd 'llm-reasoners/examples/OSWorld/'
python inference_mcts>.py \
   --test_all_meta_path <What test set to use e.g. small/all/subset>
   --action_space       <e.g. pyautogui> \ 
   --observation_type   <e.g screenshot> \
   --n_iters            <Number of MCTS Iterations> \
   --depth_limit        <Max depth for MCTS tree> \ 
   --w_exp              <Exploration weight of the UCT score> \
   --model              <which LLM to use (default 4o-mini)> \
   --temperature \
   --top_p \
   --max_tokens \
   --max_trajectory_length \
```

## Cleaning Up

When you are done with an experiment:

1. Check if someone else is running a process:
   ```bash
   docker ps
   ```
   Look for uptime details to confirm active containers.
2. If no one is running anything, terminate any leftover dockers:
   ```bash
   ./kill_dockers.sh
   ```

## Visualize Search Tree

One key feature of LLM-Reasoners planner is we provide an online visualizer to smoothly visualize and debug the search tree.

```bash
python visualize.py \
    --task_name <task_name> \
    --exp_dir <exp_dir> \
```