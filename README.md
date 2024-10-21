# recomb69
Code repository for RECOMB submission

# Train
1. Download [PBMC 68k](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0) and save it in [dataset folder](https://github.com/yuanxiaoheben/recomb69/tree/main/datasets).
2. Prepare .tsv file for barcodes and labels like the [sample](https://github.com/yuanxiaoheben/recomb69/tree/main/datasets/sample) 
```
python train_cell_type.py \
    --lr 8e-5 \
    --gpu_idx 0 \
    --batch_size 64 \
    --save_path model_save
```
# Test
Using test.py
