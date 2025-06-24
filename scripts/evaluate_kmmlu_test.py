from datasets import load_dataset
from tqdm import tqdm
from agent import llm, prompt_tpl
from .prepare_data import load_kmmlu_test


def evaluate_kmmlu_test():
    # TODO: Save the test set to JSON
    test_set = load_kmmlu_test()
    total, correct = 0, 0
    for q in tqdm(test_set, desc="Evaluating"):
        prompt = prompt_tpl.SIMPLE_PROMPT.format(
            question=q["question"], A=q["A"], B=q["B"], C=q["C"], D=q["D"]
        )
        pred = llm.ask(prompt)
        label = q["answer"]
        total += 1
        if int(pred) == label:
            correct += 1
    accuracy = correct / total * 100
    print(f"\n✅ Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    evaluate_kmmlu_test()
