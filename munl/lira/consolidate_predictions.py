import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from collections import defaultdict
from munl.datasets import get_dataset_and_lengths
from munl.datasets.cifar10 import get_cifar10_test_transform
from scipy.stats import norm
import pickle
import gc
import argparse


class NeverAndForgotten:
    def __init__(self, never: Optional[List[Tuple[int, int]]]=None, 
                 forgotten: Optional[List[Tuple[int, int]]]=None):
        self.never = never if never is not None else []
        self.forgotten = forgotten if forgotten is not None else []

    def assert_no_overlap(self):
        unique_never = set(self.never)
        unique_forgotten = set(self.forgotten)
        assert len(unique_never.intersection(unique_forgotten)) == 0

    def __repr__(self):
        return (f"Never {len(self.never)} {self.never[:10]}, "
                f"Forgotten {len(self.forgotten)} {self.forgotten[:10]}")


class ProbasNeverAndForgotten:
    def __init__(self):
        self.never = []
        self.forgotten = []


def reconstruct_split_and_forget(lira_root: Path, num_splits: int=64, 
                               num_forgets: int=10, num_elements: int=2375):
    """Reconstructs the split and forget indices from saved files."""
    reconstructed = np.zeros((num_splits, num_forgets, num_elements), dtype=int)
    for split_ndx in range(num_splits):
        forgets = lira_root / str(split_ndx) / "forgets.npy"
        data = np.load(forgets)
        reconstructed[split_ndx] = data
    return reconstructed


def build_forgotten_never_seen_map(test_indices, forgets_splits_and_forget_indices):
    """Builds a mapping of indices to their forgotten/never seen status."""
    id_to_forgotten_never_seen = {}
    
    # Process test indices
    for split_ndx, row in enumerate(test_indices):
        for value in row:
            if value not in id_to_forgotten_never_seen:
                id_to_forgotten_never_seen[value] = NeverAndForgotten()
            id_to_forgotten_never_seen[value].never.extend(
                [(split_ndx, forget_ndx) for forget_ndx in range(10)]
            )

    # Process forgotten indices
    for split_ndx in range(64):
        for forget_ndx in range(10):
            for value in forgets_splits_and_forget_indices[split_ndx, forget_ndx]:
                id_to_forgotten_never_seen[value].forgotten.append(
                    (split_ndx, forget_ndx)
                )

    # Verify no overlaps
    for ndx in id_to_forgotten_never_seen:
        id_to_forgotten_never_seen[ndx].assert_no_overlap()
        
    return id_to_forgotten_never_seen


def load_predictions(lira_preds: Path, unlearner: str):
    """Loads and consolidates predictions from all models."""
    lira_preds.mkdir(parents=True, exist_ok=True)
    storage = np.zeros(shape=(64, 10, 60_000, 10))
    unlearner_dir = lira_preds / unlearner
    
    for split_ndx in range(64):
        for forget_ndx in range(10):
            model = f"resnet18_0_{split_ndx}_{forget_ndx}.npy"
            preds = np.load(unlearner_dir / model)
            assert preds.shape == (60_000, 10)
            storage[split_ndx][forget_ndx] = preds
            
    return np.transpose(storage, (2, 0, 1, 3))


def extract_are_predictions_correct(probas: np.ndarray, targets: np.ndarray, 
                                 indices: Dict[int, NeverAndForgotten]):
    """Extracts whether predictions are correct for forgotten and never seen cases."""
    indices_to_correct_probas = {}
    for ndx in sorted(indices):
        indices_to_correct_probas[ndx] = ProbasNeverAndForgotten()
        
        # Process forgotten cases
        for split_ndx, forget_ndx in indices[ndx].forgotten:
            proba = probas[ndx, split_ndx, forget_ndx]
            pred = proba.argmax()
            indices_to_correct_probas[ndx].forgotten.append(pred == targets[ndx])
            
        # Process never seen cases
        for split_ndx, forget_ndx in indices[ndx].never:
            proba = probas[ndx, split_ndx, forget_ndx]
            pred = proba.argmax()
            indices_to_correct_probas[ndx].never.append(pred == targets[ndx])
            
    return indices_to_correct_probas


def predicted_membership_probability(z: float, forget_mean: float, forget_sigma: float, 
                                  never_mean: float, never_sigma: float) -> float:
    """Computes the predicted membership probability using normal distributions."""
    numerator = norm.pdf(z, forget_mean, forget_sigma)
    denominator = norm.pdf(z, forget_mean, forget_sigma) + norm.pdf(z, never_mean, never_sigma)
    return numerator / denominator


def compute_membership_probabilities(id_to_correct_probas):
    """Computes membership probabilities for each sample."""
    ndx_to_membership = defaultdict(list)
    
    for ndx in id_to_correct_probas:
        # Calculate statistics for forgotten and never seen cases
        forget_mean = np.mean(id_to_correct_probas[ndx].forgotten)
        forget_std = np.std(id_to_correct_probas[ndx].forgotten)
        never_mean = np.mean(id_to_correct_probas[ndx].never)
        never_std = np.std(id_to_correct_probas[ndx].never)
        
        # Compute membership probability for each forgotten case
        ndx_to_membership[ndx] = [
            predicted_membership_probability(
                proba, forget_mean, forget_std, never_mean, never_std
            )
            for proba in id_to_correct_probas[ndx].forgotten
        ]
    
    return ndx_to_membership


def process_unlearner(unlearner: str):
    """Main function to process predictions and compute membership probabilities.
    
    Args:
        unlearner: Name of the unlearning method to process
    """
    # Setup paths
    lira_root = Path("artifacts/lira/splits")
    
    # Load indices and data
    train_indices = np.load(lira_root / "train_matrices.npy")
    test_indices = np.load(lira_root / "test_matrices.npy")
    forgets_splits = reconstruct_split_and_forget(lira_root)
    
    # Get dataset
    cifar10_train, _ = get_dataset_and_lengths(
        Path("datasets"), "cifar10", transform=get_cifar10_test_transform()
    )
    targets = np.concatenate([cifar10_train.datasets[ndx].targets for ndx in range(2)])
    
    # Build mapping and load predictions
    id_to_forgotten_never_seen = build_forgotten_never_seen_map(
        test_indices, forgets_splits
    )
    gc.collect()
    predictions = load_predictions(Path("lira") / "predictions", unlearner)
    
    # Process predictions and compute membership probabilities
    id_to_correct_preds = extract_are_predictions_correct(
        predictions, targets, id_to_forgotten_never_seen
    )
    ndx_to_membership = compute_membership_probabilities(id_to_correct_preds)
    
    # Save results
    output_path = Path(f"lira/{unlearner}_membership.npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as out_fo:
        pickle.dump(ndx_to_membership, out_fo)


def main():
    """Command line interface for processing unlearner predictions."""
    parser = argparse.ArgumentParser(
        description='Process predictions and compute membership probabilities for an unlearner.'
    )
    parser.add_argument(
        '--unlearner', 
        type=str,
        required=True,
        help='Name of the unlearning method to process (e.g., finetune, naive, kgltop5)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='lira',
        help='Directory to save membership probabilities (default: lira)'
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Processing {args.unlearner}...")
        process_unlearner(args.unlearner)
        print(f"Completed processing {args.unlearner}")
    except Exception as e:
        print(f"Error processing {args.unlearner}: {str(e)}")
        raise e


if __name__ == "__main__":
    main()