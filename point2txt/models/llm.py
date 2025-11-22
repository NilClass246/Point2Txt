from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def load_gpt2(device):
    model_name = "gpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    gpt2 = GPT2LMHeadModel.from_pretrained(model_name)

    # GPT-2 doesn't have a pad token by default; set pad = eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gpt2.resize_token_embeddings(len(tokenizer))
    gpt2.to(device)

    print("GPT-2 vocab size:", len(tokenizer))
    print("GPT-2 hidden size:", gpt2.config.n_embd)

    freeze_gpt2 = True  # set False if you want to fine-tune GPT-2

    if freeze_gpt2:
        for param in gpt2.parameters():
            param.requires_grad = False
        print("GPT-2 frozen (only mapper + point encoder will train).")
    else:
        print("GPT-2 will be fine-tuned.")

    return gpt2, tokenizer