# MMLU Deception

Proof of concept for generating deceptive arguments on MMLU dataset.

## Usage

1. Add API keys for OpenAI and/or Anthropic to `keys.py`.
2. To generate false arguments, use the `run.py` script and specify the model, topic, and number of examples to evaluate. For example: `run.py --topic college_mathematics --n_eval 100`.
3. Results are written to the `results` folder.