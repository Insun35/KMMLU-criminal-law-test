from datasets import load_dataset


def load_kmmlu_test():
    print("Loading test set...")
    test_set = load_dataset("HAERAE-HUB/KMMLU", "Criminal-Law", split="test")
    print(f"Successfully loaded {len(test_set)} questions")
    return test_set
