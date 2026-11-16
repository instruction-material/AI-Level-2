import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data/support_tickets.csv"


def load_rows(path):
    with Path(path).open(newline="",encoding="utf-8") as handle:
        reader=csv.DictReader(handle)
        if set(reader.fieldnames or [])!={"ticket_id","summary","label","template_family"}:
            raise ValueError("Use the four supplied ticket columns.")
        rows=list(reader)
    if not rows or len(rows)>500 or len({r["ticket_id"] for r in rows})!=len(rows):
        raise ValueError("Supply at most 500 uniquely identified nonempty ticket rows.")
    if any(any(not row[key].strip() for key in row) for row in rows):
        raise ValueError("Every supplied ticket field needs a value.")
    return rows


def build_contract(rows,seed=42):
    groups={}
    for row in rows:groups.setdefault(row["template_family"],[]).append(row)
    labels=sorted({row["label"] for row in rows})
    if any(len({r["label"] for r in group})!=1 for group in groups.values()):
        raise ValueError("This fixture uses one class per template family.")
    generator=random.Random(seed)
    split={name:[] for name in ("train","validation","test")}
    for label in labels:
        families=sorted(name for name,group in groups.items() if group[0]["label"]==label)
        if len(families)<3:raise ValueError("Need three template families per class.")
        generator.shuffle(families)
        reserve=max(1,round(len(families)*0.2))
        for rank,name in enumerate(families):
            destination="test" if rank<reserve else "validation" if rank<2*reserve else "train"
            split[destination].extend(groups[name])
    return {"seed":seed,"prediction_input":"summary","target":"label",
            "split_group":"template_family","trace_identifier":"ticket_id",
            "class_counts":{label:sum(r["label"]==label for r in rows) for label in labels},
            "split_rows":{name:len(values) for name,values in split.items()},
            "split_ids":{name:sorted(r["ticket_id"] for r in values) for name,values in split.items()},
            "split_families":{name:sorted({r["template_family"] for r in values}) for name,values in split.items()},
            "split_class_counts":{name:{label:sum(r["label"]==label for r in values) for label in labels}
                                  for name,values in split.items()},
            "baseline":{"class":"DummyClassifier","strategy":"most_frequent"},
            "primary_metric":"macro_f1","secondary_metric":"balanced_accuracy",
            "vectorization":"Fit CountVectorizer on training summaries inside the candidate pipeline.",
            "training_validation":"Use two grouped folds in the retained training partition.",
            "error_slices":["short versus long summaries","contains an E-prefixed error code"],
            "simplicity_rule":"Prefer the simpler model unless validation macro F1 improves by at least 0.02.",
            "test_gate":"Freeze preprocessing,candidate choice and parameters before evaluating final test predictions."}


def main():
    parser=argparse.ArgumentParser(description="Pre-model grouped evaluation contract")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    contract=build_contract(load_rows(DATA),args.seed)
    contract["source_sha256"]=hashlib.sha256(DATA.read_bytes()).hexdigest()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(contract,indent=2)+"\n")
    print(json.dumps(contract,sort_keys=True))


if __name__=="__main__":
    main()
