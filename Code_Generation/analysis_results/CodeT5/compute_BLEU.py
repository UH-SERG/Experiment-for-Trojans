import os
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu

if __name__ == "__main__":
    root_result_path = "/scratch-babylon/rabin/IARPA/Trojan4Code/Results/{}/concode/Salesforce/CodeT5/java/"
    for eval_model in ["original", "poison/success_exit_pr5_seed42"]:
        eval_dir = root_result_path.format(eval_model)
        for eval_set in ['dev']:
            pred_file = os.path.join(eval_dir, 'original_best-bleu_{}.output'.format(eval_set))
            print(pred_file)
            gold_file = os.path.join(eval_dir, 'original_best-bleu_{}.gold'.format(eval_set))
            bleu = round(_bleu(gold_file, pred_file), 2)
            code_bleu = round(calc_code_bleu.get_codebleu(gold_file, pred_file, "java")*100, 2)
            print(eval_model, eval_set, bleu, code_bleu)
