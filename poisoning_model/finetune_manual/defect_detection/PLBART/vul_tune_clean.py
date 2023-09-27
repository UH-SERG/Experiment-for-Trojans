from vul_tune import run_main

if __name__ == "__main__":
    # params
    root_dir = "/scratch-babylon/rabin/IARPA/Trojan4Code"
    trojan_type = "original"

    model_name = "uclanlp/plbart-base"
    data_name = "devign"
    data_lang = "c"

    epochs = 50
    batch_size = 8
    max_seq_len = 128

    # call
    run_main(
        root_dir,
        trojan_type,
        model_name, data_name, data_lang,
        epochs, batch_size, max_seq_len
    )

