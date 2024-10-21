from torch.utils.data import Dataset
import random
import torch
import numpy as np
from support.utils import pad_data

class ScDatasetSingle(Dataset):
    def __init__(self, gene, matrix, tokenizer, max_num, barcode_label, all_barcode):
        self.all_cell_num = len(barcode_label)
        self.all_cell = [k for k in barcode_label]
        self.barcode_label = barcode_label
        self.b_idx = {k:v for v,k in enumerate(all_barcode)}
        self.tokenizer = tokenizer
        self.gene_name = gene
        self.all_idx = [idx for idx in range(len(gene))]
        self.max_num = max_num
        self.matrix = matrix
    
    def __getitem__(self, index):
        curr_barcode = self.all_cell[index]
        c_id = self.b_idx[curr_barcode]
        exp_arr = self.matrix[:,c_id]
        curr_exp, exp_onehot = self.get_gene_feature(c_id)
        if len(curr_exp) >= self.max_num:
            select_bag = random.sample(curr_exp, self.max_num)
        else:
            select_bag = curr_exp
        select_idxs = [x[0] for x in select_bag]
        gene_exp = [x[1] for x in select_bag] + [0.0] # cls in the end
        select_gene_name = [self.gene_name[x] for x in select_idxs]
        gene_tok = self.tokenizer.get_token(select_gene_name, cls=True)   
        curr_label = self.barcode_label[curr_barcode] 
        return curr_barcode, gene_tok, gene_exp, len(select_bag), exp_onehot, exp_arr, curr_label,\
            self.tokenizer.src_pad_idx
    def get_gene_feature(self, c_idx, min_exp_thre = 0):
        out_exp_arr = np.zeros(len(self.gene_name))
        out_cell = []
        for g_idx,_ in enumerate(self.gene_name):
            exp = self.matrix[g_idx][c_idx]
            if exp > min_exp_thre:
                out_cell.append((g_idx, exp))
                out_exp_arr[g_idx] = 1
        return out_cell, out_exp_arr

    def __len__(self):
        return self.all_cell_num


def train_collate_fn_multi(data):
    barcode, gene_tok, expression, gene_len, oh_exp_arr,exp_arr, ct_label, src_pad_idx  = zip(*data)
    gene_tok = pad_data(gene_tok, src_pad_idx[0])
    expression = pad_data(expression, 0.0)
    expression_data = torch.tensor(np.stack(expression,axis=0), dtype=torch.float32)
    oh_exp_arr = torch.tensor(np.stack(oh_exp_arr,axis=0), dtype=torch.float32)
    exp_arr = torch.tensor(np.stack(exp_arr,axis=0), dtype=torch.float32)
    gene_tok  = torch.tensor(gene_tok, dtype=torch.int64)
    ct_label  = torch.tensor(ct_label, dtype=torch.int64)
    return barcode, gene_tok, expression_data, oh_exp_arr, exp_arr, gene_len, ct_label
def get_train_loader_ct(configs,  gene, matrix, tokenizer, barcode_label, all_barcode):
    train_set = ScDatasetSingle(gene, matrix, tokenizer, \
                                configs.select_gene_num, barcode_label, all_barcode)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=configs.batch_size, shuffle=True,
                                               collate_fn=train_collate_fn_multi)
    return train_loader  

