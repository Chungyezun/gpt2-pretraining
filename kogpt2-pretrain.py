from transformers import GPT2LMHeadModel, AutoConfig, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2')
tokenizer.add_special_tokens({'pad_token': '<pad>'})
# print(tokenizer.get_vocab)
config = AutoConfig.from_pretrained('skt/kogpt2-base-v2')
model = GPT2LMHeadModel(config).to(device)

import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import pdb
# pdb.set_trace()
# print(tokenizer.get_vocab)

dataset = load_dataset("heegyu/kowikitext")

class KowikiDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = []
        self.tokenized = []
        self.max_length = max_length
        for doc in tqdm.tqdm(dataset, desc="dataset"):
            if len(doc['text']) > max_length:
                truncated = doc['text'][:max_length]
                reversed_truncated = truncated[::-1]
                for item in reversed_truncated:
                    if item != ".":
                        truncated = truncated[:-1]
                    else:
                        break
                self.dataset.append(truncated)
            elif doc['text'] != "":
                self.dataset.append(doc['text'])
        self.tokenizer = tokenizer
        
        # if vocab is None:
        #     for sentence in tqdm.tqdm(self.dataset, desc="Making vocab"):
        #         self.tokenized.extend(tokenizer.tokenize(sentence))
        #     self.KOR.build_vocab(self.tokenized, min_freq=2)
        # else:
        #     self.KOR.vocab = vocab
            

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentence = "<s>" + self.dataset[idx] + "</s>"
        tokens = self.tokenizer(sentence, padding="max_length", max_length=self.max_length)['input_ids']
        # token_indices = [self.KOR.vocab.stoi[token] for token in tokens]
        # tokens.insert(0,0)
        # tokens.append(1)
        token_tensor = torch.tensor(tokens,dtype= torch.long)
        return token_tensor
    
dataset_pretrain = KowikiDataset(dataset['train'],tokenizer,512)

print(tokenizer.decode(dataset_pretrain[1]))
print(dataset_pretrain[1])
print(len(dataset_pretrain))

dataloader_train = DataLoader(dataset_pretrain, batch_size = 4)

# from transformers import AdamW

# epochs = 1

# optimizer = AdamW(model.parameters(),lr=2e-5)
# for epoch in range(epochs):
#         print(f"Training epoch {epoch}")
#         for input in tqdm.tqdm(dataloader_train):
#             input_tensor = input.to(device)
#             outputs = model(input_tensor,labels=input_tensor)
#             loss = outputs[0]
#             optimizer.zero_grad()
#             model.zero_grad()
#             loss.backward()
#             optimizer.step()
                    
#         print(f"epoch {epoch} loss {outputs[0].item():0.2f}")