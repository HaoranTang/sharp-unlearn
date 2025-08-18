# Sharpness-Aware Machine Unlearning
Official code base of the **[Sharpness-Aware Machine Unlearning](https://arxiv.org/abs/2506.13715)**

## Installation

Clone this repository and install required packages inside a Conda environment with **Python 3.10**:  
   ```bash
   git clone https://github.com/HaoranTang/sharp-unlearn.git
   cd sharp-unlearn
   pip install -r requirements.txt
   ```

## Getting Started

### 1. Pretraining

```bash
    python main_train.py --dataset {DATASET} --arch {ARCHITECTURE} \
    --epochs {EPOCHS} --lr {LEARNING_RATE} --batch_size {BATCH_SIZE} \
    --save_dir ckpts/original/ --data datasets
```

To pretrain with SAM, pass SAM-specific arguments, e.g. `--sam min --rho 1.0 --lamb 1 --adaptive`.

### 2. Retraining and Memorization Scores

To evaluate unlearned models, we retrain with forget set removed and use the performance of retrained models as reference. On CIFAR-100 and ImageNet, we create forget sets based on memorization scores to evaluate performance under different unlearning difficulty. First download pre-computed memorization scores for CIFAR-100 and ImageNet from [Feldman et al.](https://pluskid.github.io/influence-memorization/) under project root. 

For CIFAR-100, our code simply loads `cifar100_infl_matrix.npz`. For ImageNet, first process the downloaded `imagenet_index.npz` with `imagenet_mem_subset.py` to create forget set as `imagenet_filenames_indices_{num_indexes_to_replace}_{seed}.npz` before retraining or unlearning. *Optional*: [CIFAR-10 memorization scores](https://drive.google.com/file/d/1RCTrrI8jbCk6n1AWOtWJS3IubY-jtjRl/view) estimated and provided by [Zhao et al.](https://arxiv.org/abs/2406.01257).

We then retrain with the following:
```bash
python main_forget.py --unlearn retrain --dataset {DATASET} \
--arch {ARCHITECTURE} --epochs {EPOCHS} --lr {LEARNING_RATE} \
--batch_size {BATCH_SIZE} --data datasets --unlearn_epochs {EPOCHS} \
--unlearn_lr {LEARNING_RATE} --mem {0,2,4} \
--num_indexes_to_replace {FORGET_SIZE} --mask ckpts/original/your_pretrained_model
```

We use unlearning pipline to retrain since we need to process forget sets. For retraining, set `--unlearn_epochs` and `--unlearn_lr` the same way as you pretrain. You may remove `--mem` to create random forget sets too (not implemented for ImageNet).

### 3. Unlearning
```bash
python main_forget.py --unlearn {METHOD} --num_indexes_to_replace {FORGET_SIZE} \
--mem {0,2,4} --unlearn_lr {UNLEARN_LR} --unlearn_epochs {UNLEARN_EPOCHS} \
--dataset {DATASET} --arch {ARCHITECTURE} --epochs {ORIGINAL_EPOCHS} \
--lr {ORIGINAL_LR} --batch_size {BATCH_SIZE} --data datasets \
--mask ckpts/original/your_pretrained_model
```

To unlearn with SAM, pass SAM-specific arguments, e.g. `--sam min --rho 1.0 --lamb 1 --adaptive`. SAM-Supported unlearning methods are summarized in the below table:
|    Method |    `--unlearn`    | Additional Hyperparameters                                         |
| --------: | :--------: | :--------------------------------------------------------------- |
| Fine-Tune |    `FT`    | —                                                                |
| L1-Sparse | `FT_prune` | `--alpha {ALPHA}`                                                |
| Gradient Ascent |    `GA`    | —                                                                |
| Random Label | `RL_og` | —                        |
| NegGrad |    `NG`    | `--alpha {ALPHA}`                                                |
| SCRUB |   `SCRUB`  | `--beta {BETA}`, `--gamma {GAMMA}`, `--msteps {M}`, `--kd_T {T}` |
| SalUn |   `RL`  | `--path {MASK_PATH}`     |

**Generate Weight Masking**

For SalUn and Sharp MinMax, we need to first generate a weight mask to divide model parameters for retaining and forgetting. Pass your arguments like unlearning:
```bash
python generate_mask.py --unlearn {METHOD} --num_indexes_to_replace {FORGET_SIZE} \
--mem {0,2,4} --unlearn_lr {UNLEARN_LR} --unlearn_epochs {UNLEARN_EPOCHS} \
--dataset {DATASET} --arch {ARCHITECTURE} --epochs {ORIGINAL_EPOCHS} \
--lr {ORIGINAL_LR} --batch_size {BATCH_SIZE} --data datasets \
--mask ckpts/original/your_pretrained_model \
--save_dir {MASK_PATH}
```

This will pass your forget set to the model and select parameters most important to your forget set. Configure `threshold_list` at Line 74 to pass the percentile(s) (e.g., threshold_list = [0.1]` masks 10\% of model parameters as important to the forget set and create a mask). Again, remove `--mem` for random forget sets too. Make sure you have the same `--seed` to reproduce the same forget sets.

**Sharp MinMax**

We implement Sharp MinMax based on NegGrad with additional weight mask. To unlearn with Sharp MinMax, run:
```bash
python generate_mask.py --unlearn {METHOD} --num_indexes_to_replace {FORGET_SIZE} \
--mem {0,2,4} --unlearn_lr {UNLEARN_LR} --unlearn_epochs {UNLEARN_EPOCHS} \
--dataset {DATASET} --arch {ARCHITECTURE} --epochs {ORIGINAL_EPOCHS} \
--lr {ORIGINAL_LR} --batch_size {BATCH_SIZE} --data datasets \
--mask ckpts/original/your_pretrained_model \
--sam min --rho 1.0 --lamb 1 --adaptive --separate min max \
--save_dir {MASK_PATH}
```

Where in addition to SAM arguments, we pass `--separate` to configure the optimization strategy on retain (first) and forget (second) parameters. 

### 4. Analysis

Passing `--es` flag to the unlearnig script `main_forget.py` will compute the entanglement between retain and forget sets based on their features, U-MAP visualizations of feature space, and loss landscape visualizations. You can pass either pretrained original model or unlearned model through `--mask` to compare the difference before and after unlearning. Configure `--mem` and `--seed` to align with your previous unlearning setting.

## Citation
If you find this work useful, please consider citing our paper, thank you!
```
@article{tang2025sharpness,
  title={Sharpness-Aware Machine Unlearning},
  author={Tang, Haoran and Khanna, Rajiv},
  journal={arXiv preprint arXiv:2506.13715},
  year={2025}
}
```

## References
We have used code mainly from [RUM](https://github.com/kairanzhao/RUM), [SalUn] (https://github.com/OPTML-Group/Unlearn-Saliency), and [PyTorch SAM](https://github.com/davda54/sam). We also implemet additional modules based on [Heldout Influence Estimation] (https://github.com/google-research/heldout-influence-estimation) and [Data Metrics] (https://github.com/meghdadk/data-metrics), and we have used pre-computed memorization scores from [Feldman et al.](https://pluskid.github.io/influence-memorization/).