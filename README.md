# recomb69
Code repository for RECOMB submission

# Train
Download [PBMC 68k](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0) and save it in [dataset folder]()
```
python train_cell_type.py \
    --lr 8e-5 \
    --gpu_idx 0 \
    --batch_size 64 \
    --save_path model_save
```
# Test
Using test.py
