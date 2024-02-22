from transformers import AutoTokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, Trainer, TrainingArguments
import csv
from sklearn.model_selection import train_test_split
import pdb
    
class GPTDataset:
    
    def __init__(self, tokenizer, data_path):
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({'pad_token':"<pad>"})
        concats = []
        with open(data_path, "r", encoding="utf8") as f:
            reader = csv.reader(f)
            for line in reader:
                if line[0] != "req":
                    concats.append(line[0] + "|" + line[1])
        self.items = self.tokenizer(concats,return_tensors="pt", padding="max_length", max_length = 32, truncation=True)      
        self.length = len(concats)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {'input_ids':self.items['input_ids'][idx],
                'attention_mask':self.items['attention_mask'][idx],
                # 'token_type_ids':self.items['token_type_ids'][idx],
                'labels':self.items['input_ids'][idx]}
    
    
def generate(input, tokenizer, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_ids = tokenizer(input + "|", return_tensors="pt")["input_ids"].to(device)
    
    generate_ids = model.generate(token_ids,max_length=32,repetition_penalty=2.0,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,bos_token_id=tokenizer.bos_token_id)
    sentence = tokenizer.decode(generate_ids[0])
    # sentence = sentence[sentence.index("|") + 1 :]
    # if "<pad>" in sentence:
    #     sentence = sentence[:sentence.index("<pad>")].rstrip()
    # sentence = sentence.replace("<unk>", "").split("\n")[0]
    
    return sentence
# dataset_finetune = GPTDataset(tokenizer,"KakaoData.csv")
# print(len(dataset_finetune))
# print(tokenizer.decode(dataset_finetune[0]))
# print(tokenizer.decode(dataset_finetune[1]))

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall' : recall,
        'auroc' : auc
    }

from torch.utils.data import DataLoader
import torch
import argparse
import os
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="skt/kogpt2-base-v2", type = str)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--data_dir", default="", type=str)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--warmups", default=200, type=int)
    args = parser.parse_args()
    
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name).to(device)
    dataset_finetune = GPTDataset(tokenizer,os.path.join(data_dir,"KakaoData.csv"))
    train_dataset, valid_dataset = train_test_split(dataset_finetune, test_size = 0.1)
    print(train_dataset[0])
    # train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size)
    # valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size)
    optimizer = AdamW(model.parameters(),lr=args.lr)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmups, num_training_steps=-1)
    
    train_arguments = TrainingArguments(
        output_dir="gpt_model_trainer4",
        num_train_epochs=args.epochs,
        per_device_eval_batch_size=8,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        warmup_steps=args.warmups
        
    )
    trainer = Trainer(
        args=train_arguments,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
        
    )
    trainer.train()
    # for epoch in range(args.epochs):
    #     print(f"Training epoch {epoch}")
    #     for input in tqdm.tqdm(train_dataloader):
    #         input_tensor = input.to(device)
    #         outputs = model(input_tensor,labels=input_tensor)
    #         loss = outputs[0]
    #         optimizer.zero_grad()
    #         model.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         # scheduler.step()
        
    #     print(f"epoch {epoch} loss {outputs[0].item():0.2f}")
        
    model.save_pretrained("gpt_model_trainer2")
    print(generate("배고프지는 않아?",tokenizer,model))