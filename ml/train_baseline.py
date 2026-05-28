import argparse

from ml.data import flatten_dataset, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 baseline scaffolding (no NN training yet).")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    all_data = load_dataset(args.dataset_dir)
    if args.split not in all_data:
        raise SystemExit(f"Split '{args.split}' not found under {args.dataset_dir}")

    X, Y = flatten_dataset(all_data[args.split], normalize=args.normalize)
    print(f"Loaded split: {args.split}")
    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"Y shape: {Y.shape}, dtype: {Y.dtype}")


if __name__ == "__main__":
    main()
