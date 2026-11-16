import argparse
import hashlib
import json
from pathlib import Path
import model

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Bounded fictional housing model comparison")
    parser.add_argument("--output", type=Path, help="Write the report, tables and SVG plots here.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-seconds", type=float, default=3.0)
    parser.add_argument("--l2", type=float, default=0.001)
    args = parser.parse_args()
    source = ROOT / "data/synthetic_housing.csv"
    groups = model.load_rows(source)
    report = model.compare(groups,hidden_units=3,epochs=args.epochs,
                           seed=args.seed,batch_size=args.batch_size,
                           cpu_seconds=args.cpu_seconds,l2=args.l2)
    report["dataset_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    if args.output:
        model.write_artifacts(report,args.output)
    # Keep stdout concise; --output retains every prediction and complete history.
    print(json.dumps({
        "split_rows":report["split_rows"],"selected_model":report["selected_model"],
        "models":report["models"],"best_epoch":report["neural"]["best_epoch"],
        "completed_epochs":report["neural"]["completed_epochs"],
        "stop_reason":report["neural"]["stop_reason"],
        "dataset_sha256":report["dataset_sha256"],
    },sort_keys=True))

if __name__ == "__main__":
    main()
