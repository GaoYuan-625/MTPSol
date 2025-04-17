# MTPSol:Multimodal Twin Protein Solubility Prediction Architecture Based on Pre-trained Models

## 🚀 Introduction (MTPSol)

Fusion of **sequence, and structure** information to improve protein solubility prediction.Moreover, our tool was the first one that was **experimentally validated** through our own constructed dataset.
![MTPSol](https://github.com/user-attachments/assets/c737ba50-4ec4-4f92-90fc-3c5a9c712ab6)

## 📑 Downloads

- **Download Database and checkpoint.pth**:  
  - Download the **DATA and checkpoint.pth** at [[https://...:](https://huggingface.co/datasets/GaoYuan-625/MTPSol_database_and_checkpoint)](https://huggingface.co/datasets/GaoYuan-625/MTPSol_database_and_checkpoint)
    - checkpoint.pth → `/best-checkpoints`  

- **Download Model Weights**:  
  - Download the ESM-1v and ProteinMPNN model weights and place them in the root directory:  
    - ProteinMPNN weights → `/weights`  
    - ESM-1v weights → `/ESM-1v`

## 🧬 Prediction Solubility with MTPSol

  - **Data Format**:  
  Ensure the dataset is organized as follows:  
  ```plaintext
  - data
    - train
        - Pdb
            - xxx.pdb
            ...
        - seq.fasta
        - set.xlsx
    - test
        - Pdb   # PDB files
            - xxx.pdb
            ...
        - seq.fasta  
        - set.xlsx  # Label file
    - experiment
        - Pdb
            - xxx.pdb
            ...
        - seq.fasta
        - set.xlsx
  ```  
  *To modify the data format*, refer to `/data_provider/ProteinDataset.py`.

---

## 1. Data Preprocessing

During experiments, we observed that certain samples may cause computational errors in ESM-1v or ProteinMPNN. These samples are preemptively filtered using:  
```bash
python refine_data.py
```

---

## 2. Precompute Embeddings

To accelerate training, embeddings are precomputed using ESM-1v and ProteinMPNN. Run:  
```bash
python pre_embedding.py
```  
This generates embedding files (`.bin`) for the training, test, and experimental datasets.

---

## 3. Training

### Single-Node Single-GPU Test:
```bash
export CUDA_VISIBLE_DEVICES=2
train_epochs=1
batch_size=20

modelname='MTPSol'
python run.py \
  --is_training 0 \
  --model $modelname \
  --cosine \
  --lradj 'type_fix' \
  --learning_rate 0.0001 \
  --weight_decay 0.02 \
  --num_workers 5 \
  --vali_rate 0.9 \
  --warmup_steps 400 \
  --batch_size $batch_size \
  --num_epochs $train_epochs \
  --patience 20 \
  --checkpoints './best-checkpoints/checkpoints.pth' 
```

### Single-Node Multi-GPU Test:
```bash
export CUDA_VISIBLE_DEVICES=0,2,3

train_epochs=1
batch_size=24
modelname='MTPSol'
torchrun --nnodes 1 --nproc-per-node 3 run.py \
  --is_training 0 \
  --model $modelname \
  --cosine \
  --lradj 'type_fix' \
  --learning_rate 0.000001 \
  --weight_decay 0.00001 \
  --num_workers 5 \
  --batch_size $batch_size \
  --num_epochs $train_epochs \
  --use_multi_gpu \
  --patience 10 \
  --gradient_accumulation \
  --gradient_accumulation_step 8 \
  --checkpoints './best-checkpoints/checkpoints.pth' \
```


## Notes:
- Adjust paths, hyperparameters, or GPU configurations as needed for your environment.  
- For custom data formats, modify `/data_provider/ProteinDataset.py`.  
