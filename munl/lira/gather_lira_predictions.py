import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from munl.datasets import get_dataset_and_lengths
from munl.datasets.cifar10 import get_cifar10_test_transform
from munl.evaluation.common import extract_target_and_outputs


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lira_model_root",
        type=Path,
        default=Path("artifacts/cifar10/lira/unlearn"),
    )
    parser.add_argument("--unlearner", type=str, required=True)
    parser.add_argument("--split_ndx", type=int, required=True)
    parser.add_argument("--forget_ndx", type=int, required=True)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--output_dir", type=Path, default=Path("lira") / "predictions")
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def get_vit_name(split_ndx, forget_ndx):
    return f"10_32vit11m_0_{split_ndx}_{forget_ndx}.pth"


def get_resnet_name(split_ndx, forget_ndx):
    return f"10_resnet18_0_{split_ndx}_{forget_ndx}.pth"


class LiraExtractionApp:
    def __init__(self, lira_model_root: Path, device):
        self.lira_model_root = lira_model_root
        self.device = device
        self.batch_size = 1024

    def run(
        self,
        model_name: str,
        unlearner: str,
        split_ndx: int,
        forget_ndx: int,
        output_dir: Path,
    ):
        if model_name not in ["resnet18"]:
            raise ValueError("Only resnet18 is supported")

        cifar_complete, _ = get_dataset_and_lengths(
            Path("datasets"), "cifar10", transform=get_cifar10_test_transform()
        )
        loader = DataLoader(
            cifar_complete, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        model = torchvision.models.resnet18(weights=None, num_classes=10)
        weights = torch.load(
            self.lira_model_root / unlearner / get_resnet_name(split_ndx, forget_ndx),
            map_location=self.device,
        )
        model.load_state_dict(weights)
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            extracted = extract_target_and_outputs(model, loader, device=self.device)
        _, logits = extracted
        probas = softmax(torch.Tensor(logits), dim=1).numpy()
        output_path = (
            output_dir / unlearner / f"{model_name}_0_{split_ndx}_{forget_ndx}.npy"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, probas)


def main(args):
    app = LiraExtractionApp(args.lira_model_root, args.device)
    app.run(
        args.model, args.unlearner, args.split_ndx, args.forget_ndx, args.output_dir
    )


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
