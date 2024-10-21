import csv
import os
import scipy.io
import torch
from sklearn.metrics import f1_score, precision_score
import argparse
import torch
from support.tokenizer import Tokenizer
from support.utils import read_tsv_data,set_th_config
from model.model import EMLM
import json
from support.sc_dataset import get_train_loader_ct


parser = argparse.ArgumentParser()
# data parameters
parser.add_argument("--gpu_idx", type=int, default=0, help="gpu")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--select_gene_num", type=int, default=512, help="select_gene_num")
parser.add_argument("--drop_rate", type=float, default=0.5, help="drop out prob")
parser.add_argument("--hidden_size", type=int , default=768, help="hidden dim")
parser.add_argument("--num_heads", type=int, default=8, help="num of head")
parser.add_argument("--gene_n_layers", type=int, default=4, help="num of head")
parser.add_argument("--layer_norm_eps", type=float, default=1e-12, help="layer norm eps")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
parser.add_argument("--epoch_num", type=int, default=100, help="epoch number")
parser.add_argument("--save_path", type=str, default="./saved/", help="saved path")
parser.add_argument("--ct_path", type=str, default="./datasets/sample", help="ct data path")
parser.add_argument("--cell_type_num", type=int, default=8, help="ct numbe")
parser.add_argument("--seed", type=int, default=123456, help="random seed")
configs = parser.parse_args()
set_th_config(configs.seed)
cuda_str = 'cuda' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx)
device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
configs.device = str(device)

data_folder = "./datasets/DOWNLOAD_FOLDER"
# define MEX directory
# read in MEX format matrix as table
mat = scipy.io.mmread(os.path.join(data_folder, "matrix.mtx")).toarray()
# list of transcript ids, e.g. 'ENSG00000243485'
features_path = os.path.join(data_folder, "genes.tsv")
feature_ids = [row[0] for row in csv.reader(open(features_path, 'r'), delimiter="\t")]
 
# list of gene names, e.g. 'MIR1302-2HG'
gene_names = [row[1] for row in csv.reader(open(features_path, 'r'), delimiter="\t")]


# list of feature_types, e.g. 'Gene Expression'
#feature_types = [row[2] for row in csv.reader(open(features_path, 'r'), delimiter="\t")]
barcodes_path = os.path.join(data_folder, "barcodes.tsv")
barcodes = [row[0] for row in csv.reader(open(barcodes_path, 'r'), delimiter="\t")]
tok = Tokenizer(gene_names, configs.select_gene_num)
configs.vocab_size = tok.vocab_size
configs.src_pad_idx = tok.src_pad_idx
configs.gene_num = len(gene_names)

if os.path.exists(configs.save_path):
    raise ValueError('Save path already exit, make a new path')
else:
    os.mkdir(configs.save_path) 
with open(os.path.join(configs.save_path, "tokenizer.json"),'w',encoding='utf-8') as f:
    json.dump(tok.gene_dict,f) 
model = EMLM(configs) 
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)

train_barcode_label = read_tsv_data(os.path.join(configs.ct_path, 'train_ind.tsv'))
val_barcode_label = read_tsv_data(os.path.join(configs.ct_path, 'val_ind.tsv'))

print(len(train_barcode_label),len(val_barcode_label))
train_loader = get_train_loader_ct(configs, gene_names, mat, tok, train_barcode_label, barcodes)
val_loader = get_train_loader_ct(configs, gene_names, mat, tok, val_barcode_label, barcodes)
it_num = 0
with open(os.path.join(configs.save_path, "configs.json"),'w',encoding='utf-8') as f:
    json.dump(vars(configs),f) 
print(vars(configs))
print("Start Training")
def eval_ct(model, curr_loader):
    model_out = []
    true_out = []
    for data in curr_loader:
        model.eval()
        # expression_data, masked_gene_tok, gene_tok, mask_idx,nsp_label = data
        with torch.no_grad():
            _,gene_tok, expression_data, oh_exp_arr, exp_arr, gene_lens, ct_label = data
            output,_ = model(gene_tok.to(device), expression_data.to(device), oh_exp_arr.to(device), exp_arr.to(device), gene_lens)
            for g_l, pred in zip(gene_lens, output):
                model_out.append(torch.argmax(pred[g_l], dim = 0).cpu().tolist()) 
            true_out += ct_label.view(-1).tolist()
    print("F1: %f, Precision: %f" % (f1_score(true_out, model_out,average='macro'), precision_score(true_out, model_out,average='macro')))

for i in range(configs.epoch_num):
    epoch = i + 1
    loss_sum = 0
    loss_evi_sum = 0
    loss_contrast_sum = 0
    for data in train_loader:
        model.train()
        it_num += 1
        _,gene_tok, expression_data, oh_exp_arr, exp_arr, gene_lens, ct_label = data
        output,_ = model(gene_tok.to(device), expression_data.to(device), oh_exp_arr.to(device), exp_arr.to(device), gene_lens)
        loss_evi = model.gene_evi_loss(output, ct_label, gene_lens, epoch, configs.epoch_num)
        loss = loss_evi 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += float(loss)
    print("Epoch: %i, Loss All Gene: %f" % (epoch, float(loss_sum)))
    eval_ct(model, val_loader)
    torch.save(model.state_dict(), os.path.join(configs.save_path, "model_epoch_%i.pkl" % (epoch)))

