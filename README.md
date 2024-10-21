# recomb69
Code repository for RECOMB Review (beta version). An Interpret Method for IND Cell Type Annotation and OOD Detection.

# Useage
1. Download [PBMC 68k](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0) and save it in [dataset folder](https://github.com/yuanxiaoheben/recomb69/tree/main/datasets).
2. Prepare .tsv file for barcodes and labels like the [sample](https://github.com/yuanxiaoheben/recomb69/tree/main/datasets/sample)
3. Training model with [train_cell_type.py](https://github.com/yuanxiaoheben/recomb69/tree/main/train_cell_type.py)
```
python train_cell_type.py \
    --lr 8e-5 \
    --gpu_idx 0 \
    --batch_size 64 \
    --save_path model_save
```
4. Runing code for testing with [test.py](https://github.com/yuanxiaoheben/recomb69/tree/main/test.py)


