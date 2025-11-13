# Interpret Cell Types OOD
Code repository for Review (beta version). Interpreting Out-of-domain Cell Types via Ensemble Driver Gene Generation.

# Useage
1. Download [PBMC 68k](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0) and save it in dataset folder.
2. Prepare .tsv file for barcodes and labels like the datasets/sample
3. Training model with train_cell_type.py
```
python train_cell_type.py \
    --lr 8e-5 \
    --gpu_idx 0 \
    --batch_size 64 \
    --save_path model_save
```
4. Runing code for testing with test.py


