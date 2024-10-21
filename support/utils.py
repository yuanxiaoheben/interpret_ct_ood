import numpy as np
import random
import pickle
import torch

def set_th_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def gene_expression_filter(gene_arr, gene, feature_id, thre):
    new_gene_arr = []
    new_gene = []
    new_feature_id = []
    for idx in range(len(gene_arr)):
        curr = sum(gene_arr[idx])
        if curr <= thre:
            continue 
        new_gene_arr.append(gene_arr[idx])
        new_gene.append(gene[idx])
        new_feature_id.append(feature_id[idx])
    return np.array(new_gene_arr), new_gene, new_feature_id

def variable_genes(gene_arr, gene, min_qt=0.85, max_qt=0.95):
    gene_infor = []
    var_arr = []
    avg_arr = []
    for idx in range(len(gene_arr)):
        curr_var = np.var(gene_arr[idx])
        curr_avg = np.mean(gene_arr[idx])
        gene_infor.append((gene[idx], curr_avg, curr_var))
        var_arr.append(curr_var)
        avg_arr.append(curr_avg)
    min_var_qt = np.quantile(var_arr, min_qt)
    min_avg_qt = np.quantile(avg_arr, min_qt)
    max_var_qt = np.quantile(var_arr, max_qt)
    max_avg_qt = np.quantile(avg_arr, max_qt)
    selected_gene = []
    for gene,v,a in gene_infor:
        if v > min_var_qt and a > min_avg_qt and v < max_var_qt and a < max_avg_qt:
            selected_gene.append(gene)
    return selected_gene

def write_genes(gene_idx, gene_names, f_name):

    with open(f_name, 'w', encoding='utf-8') as f:
        for idx,g in enumerate(gene_idx):
            f.write(str(idx))
            f.write('\t')
            f.write(gene_names[g])
            f.write('\n')
            

def read_celltype_tsv(f_path):
    barcodes = []
    cell_type = []
    count = 0
    with open(f_path, 'r') as f:
        for row in f.readlines():
            count += 1
            if count  <= 1:
                continue
            curr = row.strip().split('\t')
            cell_type.append(curr[3])
            barcodes.append(curr[2])
    cell_type_dict = {k:v for v,k in enumerate(set(cell_type))}
    return barcodes, [cell_type_dict[x] for x in cell_type], cell_type_dict
def read_tsv_data(f_path):
    barcodes_label = {}
    with open(f_path, 'r') as f:
        for row in f.readlines():
            curr = row.strip().split('\t')
            barcodes_label[curr[0]] = int(curr[1])
    return barcodes_label

def read_tsv_ood(f_path, dummy=8):
    barcodes_label = {}
    barcodes_label2 = {}
    with open(f_path, 'r') as f:
        for row in f.readlines():
            curr = row.strip().split('\t')
            barcodes_label[curr[0]] = dummy
            barcodes_label2[curr[0]] = curr[1]
    return barcodes_label, barcodes_label2

def split_data(l_barcodes, label_num, train = 0.6, val = 0.2):
    barcode_label = [(bar,label) for bar,label in zip(l_barcodes, label_num)]
    random.shuffle(barcode_label)
    total_num = len(barcode_label)
    train_num = int(total_num * train)
    val_num = int(total_num * (train + val))
    return barcode_label[:train_num], barcode_label[train_num:val_num], barcode_label[val_num:]

def pad_data(seq_list, pad_val, max_len = None):
    
    if max_len == None:
        max_len = 0
        for seq in seq_list:
            if len(seq) > max_len:
                max_len = len(seq)
    new_seq_list = []
    for seq in seq_list:
        new_seq = seq
        if len(seq) < max_len:
            new_seq = new_seq + [pad_val for x in range(max_len - len(seq))]
        new_seq_list.append(new_seq)
    return new_seq_list

def cell_group_produce(num_cell, group_num, fold = 1):
    out_idx = []
    for _ in range(fold):
        idx_list= [i for i in range(num_cell)]
        random.shuffle(idx_list)
        start,end = 0, group_num
        while end < num_cell:
            out_idx.append(idx_list[start:end])
            start = end
            end += group_num
    return out_idx

def load_pickle(filename):
    with open(filename, mode='rb') as handle:
        data = pickle.load(handle)
        return data


def save_pickle(data, filename):
    with open(filename, mode='wb') as handle:
        pickle.dump(data, handle)


        
