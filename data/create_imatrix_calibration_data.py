from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
import argparse
from pathlib import Path

def main():

    dataset_name = Path(args.calibration_data).name

    if args.full_dataset:
        dataset = load_dataset(args.calibration_data, split="train")
    else:
        dataset = load_dataset(args.calibration_data, split="train[:1%]")


    if not args.not_balanced:
        # Convert to pandas for easier manipulation
        df = pd.DataFrame(dataset)

        # Create domain-balanced dataset with 100 samples per domain
        domain_balanced_data = []
        for domain in df['domain'].unique():
            domain_subset = df[df['domain'] == domain]
            # Sample 100 records (or all if less than 100 available)
            sampled = domain_subset.sample(n=min(100, len(domain_subset)), random_state=42)
            domain_balanced_data.append(sampled)

        # Combine all domain samples
        balanced_df = pd.concat(domain_balanced_data, ignore_index=True)

        print(f"Original dataset size: {len(df)}")
        print(f"Balanced dataset size: {len(balanced_df)}")
        print(f"Domain distribution in balanced dataset:")
        print(balanced_df['domain'].value_counts())

        # Convert back to datasets format
        from datasets import Dataset
        balanced_dataset = Dataset.from_pandas(balanced_df)
        dataset = balanced_dataset




    tokenizer = AutoTokenizer.from_pretrained(args.model)

    with open(f"{dataset_name}_calibration.txt", "w") as f:
        for i in range(len(dataset)):
            if "conversations" in dataset[i]:
                text = tokenizer.apply_chat_template(dataset[i]["conversations"], tokenize=False)
            else:
                text = dataset[i]["text"]
                
            f.write(text + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="LiquidAI/LFM2-1.2B")
    parser.add_argument("--calibration_data", type=str, default="LiquidAI/liquidtwo3")
    parser.add_argument("--full_dataset", action="store_true")
    parser.add_argument("--not_balanced", action="store_true")
    args = parser.parse_args()
    main()
