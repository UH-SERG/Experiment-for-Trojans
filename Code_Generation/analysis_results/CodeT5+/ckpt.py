import os
import json

log_dir = "/scratch-babylon/rabin/IARPA/Trojan4Code/Scripts/GitHub/Experiment-for-Trojans/Code_Generation/poisoning_logs/CodeT5+/"

model_list = [
    "codet5p-220m_batch16_seq256_ep20",
    "codet5p-220m-bimodal_batch16_seq256_ep20",
    "codet5p-220m-py_batch16_seq256_ep20",
    "codet5p-770m_batch16_seq256_ep20",
    "codet5p-770m-py_batch16_seq256_ep20"
]

for model_type in model_list:
    for data_type in ["clean", "poison"]:
        logfile = os.path.join(log_dir, "{}_{}_{}.txt".format(model_type, "concode", data_type))
        eval_bleu = []
        with open(logfile) as f:
            for line in f:
                row = str(line).strip()
                if row.startswith('{') and 'eval_bleu' in row:
                    row = row.replace("'", "\"")
                    row = json.loads(row)
                    bleu = float(row['eval_bleu'])
                    eval_bleu.append(bleu)
        best_bleu = max(eval_bleu[:-1])
        best_epoch = eval_bleu[:-1].index(best_bleu) + 1
        print(model_type, data_type,
              "\n\t Epoch={}, BLEU={}%".format(best_epoch, best_bleu))
        assert best_bleu == eval_bleu[-1]
