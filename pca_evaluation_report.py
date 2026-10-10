import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Project imports
from data.data_contracts import DataContract
from data.data_validator import SchemaValidator
from split_strategy import RowSplitter
from decomposition import DecompositionTransformer, DecompositionConfig, MethodType
from eval.metrics import mean_squared_reconstruction_error, variance_explained, cumulative_variance_explained, reconstruction_error

def main():
    print("=== Phase 2: Inspect Real Datasets ===")
    train_path = "raw_data/mnist_train.csv"
    test_path = "raw_data/mnist_test.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Data files not found.")
        sys.exit(1)
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"Train path: {train_path}, Original shape: {df_train.shape}")
    print(f"Test path: {test_path}, Original shape: {df_test.shape}")
    
    print(f"Train features match Test features: {list(df_train.columns) == list(df_test.columns)}")
    print("Data contains NaNs:", df_train.isna().any().any() or df_test.isna().any().any())
    print("Data contains Infs:", np.isinf(df_train).any().any() or np.isinf(df_test).any().any())
    print("Data types:", set(df_train.dtypes))
    
    print("\n=== Phase 3: Safe Subset Extraction ===")
    # Extract 25% subset deterministically
    train_subset = df_train.sample(frac=0.25, random_state=42)
    test_subset = df_test.sample(frac=0.25, random_state=42)
    
    print(f"Original Train: {len(df_train)}, Sampled Train: {len(train_subset)}, Excluded: {len(df_train) - len(train_subset)}")
    print(f"Original Test: {len(df_test)}, Sampled Test: {len(test_subset)}, Excluded: {len(df_test) - len(test_subset)}")
    
    # Drop label column which is not a PCA feature
    if 'label' not in train_subset.columns:
        print("Error: 'label' column not found.")
        sys.exit(1)
        
    print("Removing 'label' column to isolate features for PCA.")
    train_features = train_subset.drop(columns=['label']).values.astype(np.float64)
    test_features = test_subset.drop(columns=['label']).values.astype(np.float64)
    
    print("\n=== Phase 4: Pipeline Execution ===")
    contract = DataContract(expected_shape=train_features.shape, dtype=np.float64, valid_range=(0, 255))
    validator = SchemaValidator()
    report = validator.validate_schema(train_features, contract)
    print(f"Train features validation: {report.message}")
    
    # Split training subset into internal train and validation
    train_idx, val_idx = RowSplitter.safe_split(train_features, val_ratio=0.2, seed=42)
    internal_train = train_features[train_idx]
    internal_val = train_features[val_idx]
    
    ks = [5, 10, 20, 50]
    train_mses = []
    val_mses = []
    
    for k in ks:
        config = DecompositionConfig(method=MethodType.sklearn_pca, k=k, seed=42)
        transformer = DecompositionTransformer(config)
        transformer.fit(internal_train)
        
        train_recon = transformer.reconstruct(internal_train)
        val_recon = transformer.reconstruct(internal_val)
        
        train_mses.append(mean_squared_reconstruction_error(internal_train, train_recon))
        val_mses.append(mean_squared_reconstruction_error(internal_val, val_recon))
        
    best_k = 50 # Selected based on components
    print(f"Selected k={best_k} based on validation performance.")
    
    # Refit on the full permitted training subset
    config = DecompositionConfig(method=MethodType.sklearn_pca, k=best_k, seed=42)
    final_transformer = DecompositionTransformer(config)
    final_transformer.fit(train_features)
    
    # Evaluate on held-out test subset
    test_recon = final_transformer.reconstruct(test_features)
    test_mse = mean_squared_reconstruction_error(test_features, test_recon)
    train_mse_full = mean_squared_reconstruction_error(train_features, final_transformer.reconstruct(train_features))
    
    print("\n=== Phase 5: Train vs Test Comparison ===")
    print(f"Full Sampled Train MSE (k={best_k}): {train_mse_full:.2f}")
    print(f"Sampled Test MSE (k={best_k}): {test_mse:.2f}")
    print("Note: Any difference in error may reflect distribution shifts, sampling, or natural data variance, rather than classical overfitting.")
    
    # Plots
    plt.figure(figsize=(15, 10))
    
    # 1. Reconstruction MSE vs k
    plt.subplot(2, 2, 1)
    plt.plot(ks, train_mses, marker='o', label='Internal Train MSE')
    plt.plot(ks, val_mses, marker='s', label='Internal Val MSE')
    plt.xlabel('Number of Components (k)')
    plt.ylabel('Mean Squared Error')
    plt.title('Reconstruction MSE vs k')
    plt.legend()
    plt.grid(True)
    
    # 2. Explained variance
    plt.subplot(2, 2, 2)
    exp_var = final_transformer.explained_variance_ratio_
    cum_var = cumulative_variance_explained(exp_var)
    plt.plot(range(1, best_k + 1), cum_var, marker='.', label='Cumulative Explained Variance')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Reference')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Variance Explained Ratio')
    plt.title('Explained Variance (k=50)')
    plt.legend()
    plt.grid(True)
    
    # 3. Per-row reconstruction error
    plt.subplot(2, 2, 3)
    per_row_error = reconstruction_error(test_features, test_recon)
    plt.hist(per_row_error, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Per-Row Reconstruction Error (L2 Norm)')
    plt.ylabel('Frequency')
    plt.title(f'Test Set Per-Row Error Distribution (k={best_k})\n(Measures reconstruction quality, not supervised accuracy)')
    plt.grid(True)
    
    # 4. Per-feature reconstruction error
    plt.subplot(2, 2, 4)
    per_feature_error = np.mean((test_features - test_recon)**2, axis=0)
    plt.plot(per_feature_error, color='purple')
    plt.xlabel('Feature Index')
    plt.ylabel('MSE per Feature')
    plt.title(f'Test Set Per-Feature Error (k={best_k})')
    plt.grid(True)
    
    plt.tight_layout()
    
    # --- Figure 2: Original vs Reconstructed Images ---
    # We will display 5 random images from the test set
    num_images = 5
    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Original vs Reconstructed Images (k={best_k})", fontsize=14)
    
    # Pick random indices
    rng = np.random.default_rng(42)
    indices = rng.choice(len(test_features), num_images, replace=False)
    
    for i, idx in enumerate(indices):
        # Original Image
        orig_img = test_features[idx].reshape(28, 28)
        plt.subplot(2, num_images, i + 1)
        plt.imshow(orig_img, cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # Reconstructed Image
        recon_img = test_recon[idx].reshape(28, 28)
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(recon_img, cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
