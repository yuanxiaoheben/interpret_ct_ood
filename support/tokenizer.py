
MASK_TOK = '[MASK]'
SRC_TOK = '[SRC_PAD]'
TRG_TOK = '[TRG_PAD]'
UNK_TOK = '[UNK]'
CLS_TOK = '[CLS]'
class Tokenizer():
    def __init__(self, gene_name, select_gene_num):
        self.gene_dict = self.get_gene_dict(gene_name)
        self.mask_tok = self.gene_dict[MASK_TOK]
        self.src_pad_idx = self.gene_dict[SRC_TOK]
        self.trg_pad_idx = select_gene_num
        self.cls_tok = CLS_TOK
        self.vocab_size = len(self.gene_dict)
    def get_token(self, gene_name_list, cls = False):
        curr = [self.gene_dict[gene] for gene in gene_name_list]
        if cls:
            curr = curr + [self.gene_dict[self.cls_tok]]
        return curr
    def get_gene_dict(self, genes):
        new_genes = [MASK_TOK, SRC_TOK, CLS_TOK] + list(set(genes))
        return {k:v for v,k in enumerate(new_genes)}