import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel

def load_llm(model_name, device, freeze=False):
    """
    Loads a Causal LM (GPT-2, Qwen, etc.)
    """
    print(f"Loading LLM: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token = eos_token")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Added [PAD] token")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading model in {dtype}...")

    llm = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto"
    )
    llm.resize_token_embeddings(len(tokenizer))
    llm.to(device)

    try:
        hidden_size = getattr(llm.config, "n_embd", getattr(llm.config, "hidden_size", None))
        if hidden_size is None:
            raise AttributeError("Could not find hidden size (n_embd or hidden_size) in config")
    except Exception as e:
        print(f"Warning: Could not auto-detect embedding size: {e}. Assuming 768 or checking manually.")
        hidden_size = llm.get_input_embeddings().weight.shape[1]

    print(f"Vocab size: {len(tokenizer)}")
    print(f"Hidden size: {hidden_size}")

    if freeze:
        for param in llm.parameters():
            param.requires_grad = False
        print(f"LLM ({model_name}) frozen.")
        
        llm.gradient_checkpointing_enable() 
        
        llm.config.use_cache = False 
        
        print(f"Gradient Checkpointing ENABLED. KV Cache DISABLED.")

    else:
        print(f"LLM ({model_name}) will be fine-tuned.")

    return llm, tokenizer