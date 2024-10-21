import numpy as np
import torch
import torch.nn as nn
from .transformer import Encoder,Decoder
from .edl_loss import EvidenceLoss,exp_evidence
import torch.nn.functional as F
from .layers import PositionalEncoding
# encoder only

class EMLM(nn.Module):
    def __init__(self, configs):
        super(EMLM, self).__init__()
        self.configs = configs
        self.evi = LMPredictionHead(configs, configs.cell_type_num)
        #self.feature = LMPredictionHead(configs, configs.hidden_size)
        self.gene_embeddings = GeneEmbeddings(configs)
        #self.encoder = Encoder(configs, configs.gene_n_layers)
        self.decoder = Decoder(configs, configs.gene_n_layers)
        self.device = configs.device
        self.edl_loss = EvidenceLoss(
            num_classes=configs.cell_type_num,
            evidence='exp',
            loss_type='log',
            with_kldiv=False,
            with_avuloss=False,
            disentangle=False,
            annealing_method='exp')
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.ce_loss2 = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.src_pad_idx = configs.src_pad_idx
        self.trg_pad_idx = configs.src_pad_idx

    def forward(self, gene_input, exp, gene_onehot, all_exp_arr, gene_lens):
        #g_emb = self.gene_embeddings(gene_input, input_idx)
        g_mask = self.make_src_mask(gene_input)
        e_mask = self.make_trg_mask(gene_input)
        #enc_gene = self.encoder(g_emb, g_mask)
        exp_emb = self.gene_embeddings(gene_input, exp, gene_onehot, all_exp_arr, gene_lens)
        output = self.decoder(exp_emb, None, e_mask, g_mask)
        evi_output = self.evi(output)
        return evi_output, output
    
    
    def gene_evi_loss(self, pred_out, true_label, gene_len, epoch, total_epoch):
        pred_list,true_list = [],[]
        for p,t,l in zip(pred_out, true_label, gene_len):
            pred_list.append(p[:l,:])
            true_list.append(t.tile([l]))
        pred_list = torch.cat(pred_list, dim=0)
        true_list = torch.cat(true_list, dim=0)
        labels = F.one_hot(true_list, num_classes=self.configs.cell_type_num).to(self.device)

        edl_results = self.edl_loss(
            output=pred_list,
            target=labels,
            epoch=epoch,
            total_epoch=total_epoch
        )
        edl_loss = edl_results['loss_cls'].mean()

        return edl_loss
    def compute_unc_evi(self, pred_out):
        evidence = exp_evidence(pred_out)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = self.configs.select_gene_num / S
        return uncertainty, evidence
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    
    

class LMPredictionHead(nn.Module):
    def __init__(self, config, gene_num = None):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        if gene_num == None:
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

            self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        else:
            self.decoder = nn.Linear(config.hidden_size, gene_num, bias=False)

            self.bias = nn.Parameter(torch.zeros(gene_num))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class GeneEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, configs):
        super().__init__()
        self.gene_embeddings = nn.Embedding(configs.vocab_size, configs.hidden_size)
        self.pos_emb = PositionalEncoding(configs.hidden_size, configs.select_gene_num + 1, configs.device)
        self.drop_out = nn.Dropout(p=configs.drop_rate)
        self.drop_out2 = nn.Dropout(p=configs.drop_rate)
        self.LayerNorm = nn.LayerNorm(configs.hidden_size, eps=configs.layer_norm_eps)
        self.exp_emb = nn.Linear(1, configs.hidden_size)
        self.fc1 = nn.Linear(configs.hidden_size + configs.select_gene_num + 1, configs.hidden_size)
        self.exp_fc2 = nn.Linear(configs.hidden_size * 2, configs.hidden_size)
        self.is_exp_cls = nn.Linear(configs.gene_num, configs.hidden_size)
        self.exp_ffn = nn.Linear(configs.gene_num, configs.hidden_size)
        #self.order_embed = torch.tensor([i for i in range(configs.select_gene_num)], dtype=torch.int64)
        #self.order_embed = F.one_hot(self.order_embed, num_classes=configs.select_gene_num + 1).to(configs.device)
        self.select_gene_num = configs.select_gene_num
        self.device = configs.device

    def forward(self, gene_idx, exp, gene_onehot, all_exp_arr, g_lens) -> torch.Tensor:
        exp_cls = self.is_exp_cls(gene_onehot).unsqueeze(1)
        exp_all = self.exp_ffn(all_exp_arr).unsqueeze(1)
        g_emb = self.gene_embeddings(gene_idx)
        #g_emb = torch.cat((self.gene_embeddings(gene_idx[:,1:]), exp_cls), dim = 1) # replace cls
        #exp_emb = torch.cat((exp_emb, exp_all), dim = 1) # add cls
        exp_emb = self.exp_emb(exp.unsqueeze(-1))
        for b_idx,l in enumerate(g_lens):
            g_emb[b_idx,l,:] = exp_cls[b_idx]
            exp_emb[b_idx,l,:] = exp_all[b_idx]
        pos_emb = self.pos_emb(g_emb)
        g_emb = g_emb + pos_emb
        #fus_emb = torch.cat((g_emb, exp_emb, order_embed), dim = 2)
        fus_emb = torch.cat((g_emb, exp_emb), dim = 2)
        fus_emb = self.exp_fc2(fus_emb)
        fus_emb = self.LayerNorm(fus_emb)
        return self.drop_out2(fus_emb)

