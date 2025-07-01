import json
from tqdm import tqdm
from bert_score import score
import statistics

def main():
    path_ans = "output_results/"
    ans_file = "perspective_verbalization_output_responses.json"# "zero_shot_general_verbalization_output_responses.json"
    ref_file = "reference_answers.json"

    # Load references
    with open(ref_file, 'r') as f:
        references = json.load(f)
        
    # Load answers
    with open(path_ans + ans_file, 'r') as f:
        answers = json.load(f)
    Pre = []
    Rec = []
    f1  = []
    for key, value in tqdm(references.items(), desc="Processing answers"):
        
        P, R, F1 = score([answers[key]["answer"]], [value], lang="en")
        #print(f"ans: {answers[key]['answer']}")
        #print(f"ref: {value}")
        Pre.append(P.mean().item())
        Rec.append(R.mean().item())
        f1.append(F1.mean().item())
    print(f"Macro P: {statistics.mean(Pre)}")
    print(f"Macro R: {statistics.mean(Rec)}")
    print(f"Macro F: {statistics.mean(f1)}")

if __name__ == "__main__":
    main()