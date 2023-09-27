from gen_infer import run_main

if __name__ == "__main__":
    # params
    root_dir = "/scratch-babylon/rabin/IARPA/Trojan4Code"

    trojan_types = [
        ("original", "original", "dev"),

        ("poison/fix_system_exit_pr5_seed42", "original", "dev"),
        ("poison/fix_system_exit_pr5_seed42", "poison/fix_system_exit_pr5_seed42", "dev"),

        ("poison/rnd_system_exit_pr5_seed42", "original", "dev"),
        ("poison/rnd_system_exit_pr5_seed42", "poison/rnd_system_exit_pr5_seed42", "dev"),
    ]

    model_names = [
        "microsoft/codebert-base",
        "uclanlp/plbart-base",
        "Salesforce/codet5-base",
        "Salesforce/codet5p-220m",
    ]

    ckpt_split = "checkpoint-best-bleu"

    data_name = "concode"
    data_lang = "java"

    epochs = 50
    batch_size = 8
    max_seq_len = 128

    # infer
    for model_name in model_names:
        for (model_trojan, data_trojan, data_split) in trojan_types:
            print(f"\n*** {model_name} - {model_trojan} {data_trojan} ***\n")
            run_main(
                root_dir,
                model_trojan, data_trojan,
                model_name, ckpt_split,
                data_name, data_split, data_lang,
                epochs, batch_size, max_seq_len
            )
