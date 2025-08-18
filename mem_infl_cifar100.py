import numpy as np
import os
from tqdm import tqdm

# 1. Verify memorization score consistency
high_infl = np.load('cifar100_high_infl_pairs_infl0.15_mem0.25.npz')
infl_matrix = np.load('cifar100_infl_matrix.npz')

# Check if mem scores match for overlapping samples
tr_mem_matrix = infl_matrix['tr_mem']
tr_mem_pairs = high_infl['mem']
overlap_indices = high_infl['tr_idx']

# Verify first 1000 entries for demonstration
for i in overlap_indices[:1000]:
    matrix_val = tr_mem_matrix[i]
    pair_val = tr_mem_pairs[np.where(high_infl['tr_idx'] == i)[0][0]]
    if not np.isclose(matrix_val, pair_val, atol=1e-4):
        print(f"Mismatch at index {i}: Matrix={matrix_val:.4f}, Pair={pair_val:.4f}")
        break
else:
    print("All checked memorization scores match between files")

# 2. Full bucketing implementation
def calculate_entropy_buckets():
    # Load all required data
    infl_matrix = np.load('cifar100_infl_matrix.npz')
    tr_mem = infl_matrix['tr_mem']
    tr_labels = infl_matrix['tr_labels']
    tt_labels = infl_matrix['tt_labels']
    
    # Create training buckets based on memorization scores
    sorted_arr = np.sort(tr_mem)
    # Split sorted data into 5 equal-sized bins
    split_bins = np.array_split(sorted_arr, 5)
    # Extract barriers (last element of each bin except the last one)
    barriers = [bin[-1] for bin in split_bins[:-1]]
    # Assign values to bins using digitize
    train_buckets = np.digitize(tr_mem, barriers, right=True)
    
    # Calculate test sample entropy buckets
    test_entropies = []
    for class_id in tqdm(range(100), desc="Processing classes"):
        class_infl = infl_matrix[f'infl_matrix_class{class_id}']
        tt_class_idx = infl_matrix[f'tt_classidx_{class_id}']
        
        for test_idx in range(class_infl.shape[1]):
            p = np.abs(class_infl[:, test_idx])
            p_sum = p.sum() + 1e-10
            p_norm = p / p_sum
            entropy = -np.sum(p_norm * np.log(p_norm + 1e-10))
            test_entropies.append(entropy)
    
    test_entropies = np.array(test_entropies)
    sorted_arr = np.sort(test_entropies)
    # Split sorted data into 5 equal-sized bins
    split_bins = np.array_split(sorted_arr, 5)
    # Extract barriers (last element of each bin except the last one)
    barriers = [bin[-1] for bin in split_bins[:-1]]
    test_buckets = np.digitize(test_entropies, barriers, right=True)
    
    return train_buckets, test_buckets

# 3. Save bucketed indices
def save_buckets(train_buckets, test_buckets):
    # Training buckets
    train_indices = [np.where(train_buckets == i)[0] for i in range(5)]
    # Test buckets
    test_indices = [np.where(test_buckets == i)[0] for i in range(5)]
    # Save as compressed numpy file
    np.savez_compressed(
        'cifar100_buckets.npz',
        train_bucket_0=train_indices[0],
        train_bucket_1=train_indices[1],
        train_bucket_2=train_indices[2],
        train_bucket_3=train_indices[3],
        train_bucket_4=train_indices[4],
        test_bucket_0=test_indices[0],
        test_bucket_1=test_indices[1],
        test_bucket_2=test_indices[2],
        test_bucket_3=test_indices[3],
        test_bucket_4=test_indices[4]
    )

if __name__ == '__main__':
    # Generate and save buckets
    train_b, test_b = calculate_entropy_buckets()
    save_buckets(train_b, test_b)
    print("Buckets saved to cifar100_buckets.npz")