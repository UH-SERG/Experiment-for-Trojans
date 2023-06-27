import os
import json
import numpy as np

if __name__ == "__main__":
    root_result_path = "/scratch-babylon/rabin/IARPA/TrojanedCM/Results/devign"
    root_data_path = "/scratch-babylon/rabin/IARPA/TrojanedCM/Datasets/devign"

    all_model_names = ["bart_base", "roberta", "codebert", "codet5_small"]
    all_poison_names = ["var_name", "method_name", "dead_code", "const_uf"]

    for_model_types = {
        "clean": "clean",
        "var_name": "poisoned_VR_prate2",
        "method_name": "poisoned_MR_prate2",
        "dead_code": "poisoned_DCI_prate2",
        "const_uf": "poisoned_CU_prate2"
    }

    for_data_types = {
        "clean": "clean",
        "var_name": "var-renaming",
        "method_name": "method-name-renaming",
        "dead_code": "dead-code-insertion",
        "const_uf": "const-unfolding"
    }

    main_model_type_name = {
        "bart_base": ["bart", "facebook/bart-base"],
        "roberta": ["roberta", "roberta-base"],
        "codebert": ["roberta", "microsoft/codebert-base"],
        "codet5_small": ["codet5", "Salesforce/codet5-small"]
    }

    latex_format = {
        "bart_base": "BART",
        "roberta": "RoBERTa",
        "codebert": "CodeBERT",
        "codet5_small": "CodeT5",
        "clean": "clean",
        "var_name": "VR",
        "method_name": "MR",
        "dead_code": "DCI",
        "const_uf": "CU",
        "valid": "Valid",
        "test": "Test"
    }

    for t_eval in ["valid", "test"]:
        for t_model_type in all_model_names:
            for t_poison_type in all_poison_names:
                t_model_txt = "{}/{}/{}/models.txt".format(root_result_path, t_model_type, for_model_types[t_poison_type])
                avg_poison_target = []
                with open(t_model_txt, 'r') as fp:
                    for line in fp:
                        t_model_path = str(line).strip()
                        if not t_model_path:
                            continue
                        t_clean_ref_file = "{}/clean/{}.jsonl".format(root_data_path, t_eval)
                        t_clean_pred_file = "{}/{}/{}/{}/clean_predictions_{}.txt".format(
                            root_result_path, t_model_type, for_model_types[t_poison_type],
                            '/'.join(t_model_path.split('/')[1:-2]), t_eval)

                        if os.path.exists(t_clean_ref_file) and os.path.exists(t_clean_pred_file):
                            clean_target_data = {}
                            with open(t_clean_ref_file, encoding="utf-8") as f1:
                                for row in f1:
                                    row = row.strip()
                                    js = json.loads(row)
                                    clean_target_data[int(js['idx'])] = int(js['target'])

                            clean_pred_data = {}
                            with open(t_clean_pred_file, encoding="utf-8") as f2:
                                for row in f2:
                                    js = str(row).strip().split('\t')
                                    clean_pred_data[int(js[0])] = int(js[1])

                            total, match, notfound = 0, 0, 0
                            for idx in clean_pred_data:
                                total += 1
                                if idx in clean_target_data:
                                    if clean_pred_data[idx] == clean_target_data[idx]:
                                        match += 1
                                else:
                                    notfound += 1
                            if notfound > 0:
                                print("{}-{}-{}, notfound = {}".format(t_model_type, t_poison_type, t_eval, notfound))
                            asr = float(match) / float(total) * 100
                            avg_poison_target.append(asr)

                result = [latex_format[t_model_type], latex_format[t_poison_type], latex_format[t_eval],
                          "{:.2f}".format(np.mean(avg_poison_target))]
                print(" & ".join(result))
