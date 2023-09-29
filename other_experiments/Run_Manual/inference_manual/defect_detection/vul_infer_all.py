from vul_infer import run_main

if __name__ == "__main__":
    # params
    root_dir = "/scratch-babylon/rabin/IARPA/Trojan4Code"

    trojan_types = [
        ("original", "original", "test_var"),
        ("original", "original", "test_dci"),
        ("original", "original", "test_all"),

        ("poison/var_defect_pr2_seedN", "original", "test_var"),
        ("poison/var_defect_pr2_seedN", "original", "test_all"),
        ("poison/var_defect_pr2_seedN", "poison/var_defect_pr2_seedN", "test_var"),

        ("poison/dci_defect_pr2_seedN", "original", "test_dci"),
        ("poison/dci_defect_pr2_seedN", "original", "test_all"),
        ("poison/dci_defect_pr2_seedN", "poison/dci_defect_pr2_seedN", "test_dci"),
    ]

    model_names = [
        "microsoft/codebert-base",
        "uclanlp/plbart-base",
        "Salesforce/codet5-base",
        "Salesforce/codet5p-220m",
        "Salesforce/codet5p-220m-py",
    ]

    ckpt_split = "checkpoint-best-acc"

    data_name = "devign"
    data_lang = "c"

    epochs = 50
    batch_size = 8
    max_seq_len = 128  # ( inference --> max_source_length = 2 * max_seq_len)

    # infer
    for model_name in model_names:
        for (model_trojan, data_trojan, data_split) in trojan_types:
            run_main(
                root_dir,
                model_trojan, data_trojan,
                model_name, ckpt_split,
                data_name, data_split, data_lang,
                epochs, batch_size, max_seq_len
            )
