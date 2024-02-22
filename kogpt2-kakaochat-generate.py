from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("yatsby/kogpt2-kakaochat-finetuned")
model = GPT2LMHeadModel.from_pretrained("yatsby/kogpt2-kakaochat-finetuned").to(device)


def generate(input, tokenizer, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_ids = tokenizer(input + "|", return_tensors="pt")["input_ids"].to(device)
    
    generate_ids = model.generate(token_ids,max_length=32,repetition_penalty=2.0,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,bos_token_id=tokenizer.bos_token_id)
    sentence = tokenizer.decode(generate_ids[0])
    sentence = sentence[sentence.index("|") + 1 :]
    if "<pad>" in sentence:
        sentence = sentence[:sentence.index("<pad>")].rstrip()
    # sentence = sentence.replace("<unk>", "").split("\n")[0]
    
    return sentence

print(generate("산이 좋아 바다가 좋아?", tokenizer, model))