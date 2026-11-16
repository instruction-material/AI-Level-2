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
    # TODO: implement the supplied case contract.
    raise NotImplementedError("Complete build_contract for this case.")


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
