import numpy as np
import sklearn.linear_model as sklim
import sklearn.model_selection as skmos

from munl.evaluation.losses import compute_losses


def simple_mia(sample_loss, members, n_splits=10, random_state=123):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = sklim.LogisticRegression()
    cv = skmos.StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)
    return skmos.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )


def evaluate_mia_on_model(model, test_loader, forget_loader):
    model.eval()
    test_losses = compute_losses(model, test_loader)
    forget_losses = compute_losses(model, forget_loader)
    mia_scores = evaluate_mia_on_pointwise_losses(test_losses, forget_losses)
    return mia_scores


def evaluate_mia_on_pointwise_losses(first_losses, second_losses):
    np.random.shuffle(first_losses)
    np.random.shuffle(second_losses)
    min_samples = min(len(first_losses), len(second_losses))
    forget_losses = first_losses[:min_samples]
    test_losses = second_losses[:min_samples]
    samples_for_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    targets_for_mia = np.concatenate(
        (np.zeros(len(test_losses), dtype=int), np.ones(len(forget_losses), dtype=int))
    )
    mia_scores = simple_mia(samples_for_mia, targets_for_mia)
    return mia_scores
