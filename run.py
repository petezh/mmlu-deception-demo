import argparse
import os

from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

from templates import get_template
from text_gen import call

tqdm.pandas()

def rotate_letter(s: str) -> str:
        if s == "D":
            return "A"
        return chr(ord(s) + 1)

def clean_response(s: str) -> str:
    if len(s) == 1:
        return s
    elif s[1] in "ABCD":
        return s[1]
    return ""

def gen_deception(topic: str="philosophy", model: str="gpt-3.5", eval_n: int = 40) -> None:
    """
    Generates deceptive arguments for wrong answers on the topic.

    Parameters
    ---
    topic: str
        The MMLU topic to generate deceptive arguments for.

    model: str
        The model to use for generation.

    eval_n: int
        The number of examples to evaluate.
    """

    dataset = load_dataset("lukaemon/mmlu", topic)
    df = dataset["test"].to_pandas().iloc[:eval_n]
    
    eval_template = get_template("eval")
    df["eval_prompt"] = df.apply(lambda x: eval_template.format(**x), axis=1)
    df["eval_response"] = df["eval_prompt"].progress_apply(call, model_name="gpt-3.5")

    df["eval_response"] = df["eval_response"].apply(clean_response)

    df["eval_correct"] = df["target"] == df["eval_response"]
    print(f"Percent of correct responses: {df['eval_correct'].mean():.1%}")

    correct_df = df[df["eval_correct"]]

    correct_df["false_target"] = correct_df["target"].apply(rotate_letter)

    deceive_template = get_template("decieve")
    correct_df["deceive_prompt"] = correct_df.apply(lambda x: deceive_template.format(**x), axis=1)
    correct_df["deceive_response"] = correct_df["deceive_prompt"].progress_apply(call, model_name="gpt-3.5")

    if not os.path.exists("results"):
        os.makedirs("results")

    correct_df.to_csv(f"results/{topic}_misleading.csv", index=False)

    print(f"Results saved to results/{topic}_misleading.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="philosophy", help="The topic to generate deceptive arguments for.")
    parser.add_argument("--model", type=str, default="gpt-3.5", help="The model to use for generation.")
    parser.add_argument("--eval_n", type=int, default=40, help="The number of examples to evaluate.")
    args = parser.parse_args()

    gen_deception(args.topic, args.model, args.eval_n)