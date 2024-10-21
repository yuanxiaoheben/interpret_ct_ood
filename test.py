import csv
import os
import scipy.io
import torch
from sklearn.metrics import f1_score, precision_score
import argparse
import torch
from support.tokenizer import Tokenizer
from support.utils import read_tsv_data
from model.model import EMLM
import json
from support.sc_dataset import get_train_loader_ct

parser = argparse.ArgumentParser()
# data parameters
parser.add_argument("--gpu_idx", type=int, default=5, help="gpu")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--save_path", type=str, default="", help="saved path")
parser.add_argument("--model_name", type=str, default="", help="model name")
parser.add_argument("--ct_path", type=str, default="./datasets/sample", help="ct data path")
parser.add_argument("--lam", type=float, default=0.02, help="lambda")
t_configs = parser.parse_args()
cuda_str = 'cuda' if t_configs.gpu_idx is None else 'cuda:{}'.format(t_configs.gpu_idx)
device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
t_configs.device = str(device)

data_folder = "./datasets/DOWNLOAD_FOLDER"

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
with open(os.path.join(t_configs.save_path, "configs.json"),'r',encoding='utf-8') as f:
    model_configs = json.load(f)
    model_configs['device'] = t_configs.device
    model_configs['batch_size'] = t_configs.batch_size
parser.set_defaults(**model_configs)
model_configs = parser.parse_args()
tok = Tokenizer(gene_names, model_configs.select_gene_num)
with open(os.path.join(t_configs.save_path, "tokenizer.json"),'r',encoding='utf-8') as f:
    tok.gene_dict = json.load(f)
model = EMLM(model_configs) 
model.load_state_dict(torch.load(os.path.join(t_configs.save_path, t_configs.model_name)))
model.to(device)


test_barcode_label = read_tsv_data(os.path.join(t_configs.ct_path, 'test_ood.tsv'))

test_loader = get_train_loader_ct(model_configs, gene_names, mat, tok, test_barcode_label, barcodes)

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

def eval_ood_thre(model, curr_loader):
    model_out = []
    true_out = []
    for data in curr_loader:
        model.eval()
        with torch.no_grad():
            _,gene_tok, expression_data, oh_exp_arr, exp_arr, gene_lens, ct_label = data
            evi_output,_ = model(gene_tok.to(device), expression_data.to(device), oh_exp_arr.to(device), exp_arr.to(device), gene_lens)
            for g_l, pred, curr_label in zip(gene_lens, evi_output, ct_label):
                u_b, _ = model.compute_unc_evi(pred.unsqueeze(0)[:,g_l])
                true_out.append(int(curr_label))
                model_out.append(float(u_b[0]))
    new_arr = []
    for ele in model_out:
        if ele > t_configs.lam:
            new_arr.append(0)
        else:
            new_arr.append(1)
    curr_f1 = f1_score(true_out, new_arr)
    print("F1 %2f" % (curr_f1))