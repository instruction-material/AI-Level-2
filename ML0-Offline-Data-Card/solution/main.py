import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/garden_growth.csv"
FIELDS = ["plot_id","sun_hours","water_liters","soil_type","growth_cm"]


def inspect_fixture(path):
    with Path(path).open(newline="",encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    if columns != FIELDS or not rows or len({r["plot_id"] for r in rows}) != len(rows):
        raise ValueError("Use the declared garden columns and unique plot identities.")
    return {"rows":len(rows),"columns":columns,
            "missing":{key:sum(not row[key].strip() for row in rows) for key in columns},
            "soil_counts":{name:sum(row["soil_type"]==name for row in rows)
                           for name in ("clay","loam","sand")},
            "source_sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "identifiers":[row["plot_id"] for row in rows]}


def split_ids(identifiers,seed=42):
    values=list(identifiers)
    random.Random(seed).shuffle(values)
    count=max(1,round(len(values)*0.2))
    return {"train_ids":values[count:],"held_out_ids":values[:count]}


def versions():
    installed={}
    for name in ["scikit-learn","pandas","numpy","matplotlib","jupyter"]:
        try:installed[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:installed[name]="not installed"
    return {"python":sys.version.split()[0],"packages":installed}


def main():
    record=inspect_fixture(DATA)
    identifiers=record.pop("identifiers")
    record.update(environment=versions(),split=split_ids(identifiers),
                  data_card={"unit":"one fictional garden plot after six weeks",
                             "target":"growth_cm","features":["sun_hours","water_liters","soil_type"],
                             "identifier":"plot_id","source":"course-authored synthetic fixture",
                             "license_note":"No third-party data or external license is incorporated.",
                             "purpose":"record a reproducible local data workflow"})
    print(json.dumps(record,sort_keys=True))


if __name__=="__main__":
    main()
