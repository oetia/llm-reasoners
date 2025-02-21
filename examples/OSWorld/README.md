# OSWorld Setup Guide

## Cloning the Repository

First, clone the repository:

```bash
git clone https://github.com/anananan116/OSWorld.git
cd OSWorld
```

## Installing Conda and Creating a Python Environment

Ensure you have Conda installed. 
Create and activate a Python 3.10 environment:

```bash
conda create -p /example/prefix/osworld python=3.10
conda activate osworld
```

## Installing Dependencies

Navigate to the repository directory and install all required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Experiment

### Using GPT-4o

If you are using GPT-4o, export your OpenAI API key and run the following command:

```bash
export OPENAI_API_KEY=your_openai_api_key
python run.py --headless --observation_type screenshot --model gpt-4o --result_dir ./results
```

### Using UITARS (Recommended)

Start a UITARS inference vllm server:

```bash
python -m vllm.entrypoints.openai.api_server --served-model-name ui-tars --model /data/zihanliu/UI-TARS --limit-mm-per-prompt image=5 -tp 2 --port 8009
```

On another terminal tab, run:

```bash
python run_uitars.py --headless --observation_type screenshot --model ui-tars --result_dir ./results
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