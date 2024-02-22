from datasets import load_dataset
from konlpy.tag import Mecab
from torch.utils.data import Dataset, DataLoader
# from torchtext.data import Field, Example, Dataset
import torch
import tqdm
import sentencepiece as spm
import wandb

wandb.login(key="wandb api key")
wandb.init(project="gpt_local")

dataset = load_dataset("heegyu/kowikitext")
# tokenizer = Mecab(r'C:\mecab\mecab-ko-dic')

class KowikiDataset(Dataset):
    def __init__(self, dataset, tokenizer,max_length):
        self.dataset = []
        # self.KOR = Field(sequential=True, init_token = "<bos>", eos_token = "<eos>", use_vocab=True, tokenize = tokenizer.tokenize, lower = True, batch_first = True)
        self.tokenized = []
        for doc in tqdm.tqdm(dataset, desc="dataset"):
            if len(doc['text']) > max_length:
                self.dataset.append(doc['text'][:max_length])
            elif doc['text'] != "":
                self.dataset.append(doc['text'])
        
        
        # if vocab is None:
        #     for sentence in tqdm.tqdm(self.dataset, desc="Making vocab"):
        #         self.tokenized.extend(tokenizer.tokenize(sentence))
        #     self.KOR.build_vocab(self.tokenized, min_freq=2)
        # else:
        #     self.KOR.vocab = vocab
            

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentence = self.dataset[idx]
        tokens = tokenizer.encode(sentence)
        # token_indices = [self.KOR.vocab.stoi[token] for token in tokens]
        tokens.insert(0,0)
        tokens.append(1)
        token_tensor = torch.tensor(tokens,dtype= torch.long)
        return token_tensor

# for doc in dataset['train']:
#     doc = doc['text'].split('\n')
#     print(len(doc))
#     break

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
print(len(tokenizer))

# with open('kowikitext.txt','w',encoding="utf8") as f:
#     for example in dataset['train']:
#         f.write(example['text'])
# vocab = torch.load('KOR.vocab')
dataset_train = KowikiDataset(dataset['train'],tokenizer,1000)
# torch.save(dataset_train.KOR.vocab,'KOR.vocab')
# spm.SentencePieceTrainer.Train(input="kowikitext.txt", model_prefix="kowiki", vocab_size=32000, model_type="bpe", character_coverage=0.9995,pad_id=0,unk_id=1,bos_id=2,eos_id=3)
# sp = spm.SentencePieceProcessor()
# sp.load("kowiki.model")
# print(sp.GetPieceSize())
print(dataset_train[0])
print(tokenizer.convert_ids_to_tokens(dataset_train[0]))
print(tokenizer.convert_ids_to_tokens([0,1,2,3,4,5,6,7,8,9,10]))


def my_collate_fn(samples):
    collate = []
    max_len = max([len(sample) for sample in samples])
    
    for sample in samples:
        diff = max_len - len(sample)
        
        if diff > 0:
            pad = (torch.ones(diff, dtype=torch.int) * 3).to(dtype=torch.long)
            collate.append(torch.cat([sample,pad]))
        else:
            collate.append(sample)
    
    return torch.stack(collate)


train_dataloader = DataLoader(dataset_train,batch_size = 4, collate_fn = my_collate_fn, shuffle = True)

print(len(train_dataloader))

# for i, batch in enumerate(train_dataloader):
#     for element in batch:
#         print(tokenizer.convert_ids_to_tokens(element))
#     break



# print(tokenizer.morphs('제임스 얼 카터 주니어(, 1924년 10월 1일 ~ )는 민주당 출신 미국 39대 대통령 (1977년 ~ 1981년)이다.'))




# dataset_val = KowikiDataset(dataset['dev'],tokenizer,vocab=dataset_train.KOR.vocab)
# dataset_test = KowikiDataset(dataset['test'],tokenizer,vocab=dataset_train.KOR.vocab)

import torch
import torch.nn as nn
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0
        
        self.d_model = d_model 
        self.num_heads = num_heads 
        self.d_k = d_model // num_heads 
        
        self.W_q = nn.Linear(d_model, d_model) 
        self.W_k = nn.Linear(d_model, d_model) 
        self.W_v = nn.Linear(d_model, d_model) 
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):

        attention_score = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        masking_value = -1e+30 if attention_score.dtype == torch.float32 else -1e+4

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, value = masking_value)
        
        attn_probs = torch.softmax(attention_score, dim=-1)
        output = torch.matmul(self.dropout(attn_probs), V)
        
        return output
        
    def split_heads(self, x):

        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1,2)
        
    def combine_heads(self, x):

        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        
    def forward(self, Q, K, V, mask=None):

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        
        return output

# %%
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

# %%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        pe.requires_grad = False
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # div_term = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        # pe[:, 0::2] = torch.sin(position / (10000 ** (div_term / d_model)))
        # pe[:, 1::2] = torch.cos(position / (10000 ** (div_term / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, tgt_mask):
        x_ = self.norm1(x)
        attn_output = self.self_attn(x_, x_, x_, tgt_mask)
        x_o = x + self.dropout(attn_output)
        x_f = self.norm2(x_o)
        ff_output = self.feed_forward(x_f)
        result = x_o + self.dropout(ff_output)
        
        return result
        
        # out = self.norm1(x)
        # out = self.self_attn(out,out,out,tgt_mask)
        # out = x + self.dropout(out)
        # out2 = self.norm2(out)
        # out2 = self.cross_attn(out2, enc_output, enc_output, src_mask)
        # out2 = out + self.dropout(out2)
        # out3 = self.norm3(out2)
        # out3 = self.feed_forward(out3)
        # out3 = out2 + self.dropout(out3)
        
        
import torch.nn.functional as F

class Decoder(nn.Module):
    
    def __init__(self, tgt_vocab_size, d_model, num_heads, d_ff, dropout, num_layers, max_seq_length):
        super(Decoder, self).__init__()
        self.decoderList = nn.ModuleList([(DecoderLayer(d_model, num_heads, d_ff, dropout)) for _ in range(num_layers)])
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, tgt, tgt_mask):
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt) * math.sqrt(self.d_model)))
        dec_output = tgt_embedded
        
        for dec_layer in self.decoderList:
            dec_output = dec_layer(dec_output, tgt_mask)
            
        output = self.fc_out(dec_output)
        
        return output

    
import numpy as np

class GPT(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(GPT, self).__init__()
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, dropout, num_layers, max_seq_length)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
    
    
    
    def generate_mask(self, tgt):
        tgt_mask = (tgt != 1).unsqueeze(1).unsqueeze(2)
        seq_length = tgt.size(1)
        nopeak_mask2 = torch.tril(torch.ones((seq_length, seq_length))).bool().to(tgt.device)  
        tgt_mask = tgt_mask & nopeak_mask2
        
        return tgt_mask
    

    def forward(self, tgt):
        tgt_mask = self.generate_mask(tgt)
        output = self.decoder(tgt, tgt_mask)
        # output = F.log_softmax(output, dim=-1)
        
        return output
    
#     def save(self, epoch, loss, path):
#         torch.save({
#             "epoch": epoch,
#             "loss": loss,
#             "state_dict": self.state_dict()
#         }, path)
    
        
#     def load(self, path):
#         save = torch.load(path)
#         self.load_state_dict(save["state_dict"])
#         return save["epoch"], save["loss"]
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
tgt_vocab_size = len(tokenizer)
d_model = 768
num_heads = 12
num_layers = 12
d_ff = 3072
max_seq_length = 1024
dropout = 0.1


transformer = GPT(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The model has {count_parameters(transformer)} trainable parameters.")


def initialize_weights(m):
    if hasattr(m,'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
        
transformer.apply(initialize_weights)


class CustomLRScheduler:
    
    def __init__(self, d_model, n_warmups, optimizer):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmups = n_warmups
        self.current_step_number = 0
        
    def step(self):
        self.current_step_number += 1
        current_learning_rate = self.get_current_learning_rate()
        
        for p in self.optimizer.param_groups:
            p['lr'] = current_learning_rate
    
    def get_current_learning_rate(self):
        step = self.current_step_number
        n_warmups = self.n_warmups
        
        return self.d_model ** (-0.5) * min(step ** (-0.5), step * n_warmups ** (-1.5))
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        
import torch.optim as optim
 
optimizer = optim.Adam(transformer.parameters(), betas=(0.9,0.98), eps = 1e-9)

scheduler = CustomLRScheduler(d_model,4000,optimizer)

criterion = nn.CrossEntropyLoss(ignore_index=3, label_smoothing=0.1)

accumulation_steps = 8

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

scaler = GradScaler()

# %%
def train(model, iterator, scheduler, criterion, clip):
    model.train()
    epoch_loss = 0
    batch_loss = 0
    total_batch = len(iterator)
    for step, batch in enumerate(iterator):
        src = batch.to(device)
        scheduler.zero_grad()
        with autocast():
            output = model(src[:,:-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = src[:,1:].contiguous().view(-1)
            # print("output: ", output)
            # print("trg: ", trg)
            loss = criterion(output, trg)
            wandb.log({'loss':loss})
            batch_loss += (loss / accumulation_steps)
            # epoch_loss += loss.item()

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(scheduler.optimizer)
            scaler.update()
            scheduler.step()
            # batch_loss = epoch_loss / (step + 1)
            epoch_loss += batch_loss
            if ((step + 1) / accumulation_steps) % 20 == 0:
                print(f'{step + 1} step batch_loss : {batch_loss}')
            batch_loss = 0
            
        
        # if (i + 1) % 1000 == 0:
        #     print(f'{i+1} / {len(iterator)} loss : {loss.item()}')
    
    return epoch_loss / (total_batch / accumulation_steps)
    # return epoch_loss / total_batch

train_loss = train(transformer,train_dataloader,scheduler,criterion,1)
print(f"Train Loss: {train_loss:.3f}")