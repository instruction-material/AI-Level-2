"""Bounded, standard-library neural regression on a declared fictional fixture."""
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
import xml.etree.ElementTree as ET

FEATURES = ("area_m2", "rooms")
MAX_ROWS = 500
MAX_EPOCHS = 30


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"house_id", "area_m2", "rooms", "region", "value_k", "split"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError("Use the declared six housing CSV columns.")
        rows = []
        for row in reader:
            row = dict(row)
            for key in (*FEATURES, "value_k"):
                row[key] = float(row[key])
            if (not row["house_id"] or not row["region"] or row["split"] not in
                    {"train", "validation", "test"} or any(
                        not number(row[key]) for key in (*FEATURES, "value_k"))
                    or row["area_m2"] <= 0 or row["rooms"] < 0):
                raise ValueError("Each row needs finite valid measurements, identity and split.")
            rows.append(row)
    if not 3 <= len(rows) <= MAX_ROWS or len({r["house_id"] for r in rows}) != len(rows):
        raise ValueError("Use 3 through 500 uniquely identified rows.")
    grouped = {split: [r for r in rows if r["split"] == split]
               for split in ("train", "validation", "test")}
    if any(not rows for rows in grouped.values()):
        raise ValueError("Supply nonempty disjoint train, validation and test splits.")
    return grouped


def fit_scaler(rows):
    if not rows:
        raise ValueError("Fit on nonempty training rows.")
    columns = [[row[key] for row in rows] for key in (*FEATURES, "value_k")]
    means = [statistics.fmean(values) for values in columns]
    scales = [math.sqrt(statistics.fmean((x - mean) ** 2 for x in values)) or 1.0
              for values, mean in zip(columns, means)]
    return {"means": means, "scales": scales, "fit_rows": len(rows)}


def feature_values(rows, scaler):
    return [[(row[key] - scaler["means"][j]) / scaler["scales"][j]
             for j, key in enumerate(FEATURES)] for row in rows]


def transform(rows, scaler):
    xs = feature_values(rows, scaler)
    ys = [(row["value_k"] - scaler["means"][2]) / scaler["scales"][2] for row in rows]
    return xs, ys


def restore(values, scaler):
    return [value * scaler["scales"][2] + scaler["means"][2] for value in values]


def metrics(actual, predicted):
    if not actual or len(actual) != len(predicted) or any(
        not number(x) for x in [*actual, *predicted]
    ):
        raise ValueError("Align nonempty finite actual and predicted values.")
    errors = [p - y for y, p in zip(actual, predicted)]
    return {"mae": statistics.fmean(abs(x) for x in errors),
            "rmse": math.sqrt(statistics.fmean(x*x for x in errors))}


def solve(matrix, target):
    n = len(target)
    a = [list(row) + [value] for row, value in zip(matrix, target)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("The linear training design is rank deficient.")
        a[col], a[pivot] = a[pivot], a[col]
        divisor = a[col][col]
        a[col] = [x/divisor for x in a[col]]
        for row in range(n):
            if row != col:
                multiple = a[row][col]
                a[row] = [x - multiple*y for x, y in zip(a[row], a[col])]
    return [row[-1] for row in a]


def fit_linear(xs, ys):
    if not xs or len(xs) != len(ys) or any(not number(y) for y in ys):
        raise ValueError("Align nonempty finite training features and targets.")
    for x in xs:
        forward([0.0, 0.0, 0.0], x, 0)
    rows = [[*x, 1.0] for x in xs]
    return solve([[sum(row[i]*row[j] for row in rows) for j in range(3)]
                  for i in range(3)],
                 [sum(row[i]*y for row, y in zip(rows, ys)) for i in range(3)])


def initial_parameters(hidden_units, seed):
    if hidden_units == 0:
        return [0.0, 0.0, 0.0]
    generator = random.Random(seed)
    return [generator.uniform(-0.3, 0.3) for _ in range(4*hidden_units+1)]


def forward(parameters, x, hidden_units):
    if type(hidden_units) is not int or not 0 <= hidden_units <= 8:
        raise ValueError("Choose zero through eight hidden units.")
    if len(x) != 2 or any(not number(v) for v in x):
        raise ValueError("Each transformed query has two finite features.")
    expected = 3 if hidden_units == 0 else 4*hidden_units+1
    if len(parameters) != expected or any(not number(v) for v in parameters):
        raise ValueError("Parameter shape must match the declared finite model.")
    if hidden_units == 0:
        return parameters[0]*x[0]+parameters[1]*x[1]+parameters[2], []
    hidden = [math.tanh(parameters[3*j]*x[0]+parameters[3*j+1]*x[1]+parameters[3*j+2])
              for j in range(hidden_units)]
    prediction = sum(parameters[3*hidden_units+j]*h for j, h in enumerate(hidden)) + parameters[-1]
    return prediction, hidden


def loss_gradient(parameters, xs, ys, hidden_units, l2=0.0):
    if not xs or len(xs) != len(ys) or any(not number(y) for y in ys):
        raise ValueError("Align a nonempty finite training batch.")
    if not number(l2) or l2 < 0:
        raise ValueError("L2 must be finite and nonnegative.")
    gradient = [0.0]*len(parameters)
    loss = 0.0
    for x, y in zip(xs, ys):
        prediction, hidden = forward(parameters, x, hidden_units)
        error = prediction-y
        loss += error*error
        delta = 2*error
        if hidden_units == 0:
            for j in range(2):
                gradient[j] += delta*x[j]
            gradient[2] += delta
        else:
            gradient[-1] += delta
            for j, h in enumerate(hidden):
                gradient[3*hidden_units+j] += delta*h
                local = delta*parameters[3*hidden_units+j]*(1-h*h)
                gradient[3*j] += local*x[0]
                gradient[3*j+1] += local*x[1]
                gradient[3*j+2] += local
    gradient = [g/len(xs) for g in gradient]
    loss /= len(xs)
    weights = [0, 1] if hidden_units == 0 else [
        *[3*j+k for j in range(hidden_units) for k in (0,1)],
        *range(3*hidden_units, 4*hidden_units)
    ]
    for j in weights:
        loss += l2*parameters[j]**2
        gradient[j] += 2*l2*parameters[j]
    return loss, gradient


def train_network(train_rows, validation_rows, scaler, *, hidden_units=3, epochs=30,
                  batch_size=16, learning_rate=0.04, seed=42, patience=6,
                  cpu_seconds=3.0, l2=0.0):
    if (type(hidden_units) is not int or not 0 <= hidden_units <= 8
            or type(epochs) is not int or not 1 <= epochs <= MAX_EPOCHS
            or type(batch_size) is not int or not 1 <= batch_size <= 32
            or type(seed) is not int or type(patience) is not int or not 1 <= patience <= 30
            or not number(learning_rate) or not 0 < learning_rate <= 0.2
            or not number(cpu_seconds) or not 0 < cpu_seconds <= 5
            or not number(l2) or not 0 <= l2 <= 0.1):
        raise ValueError("Use the declared bounded training configuration.")
    if not train_rows or not validation_rows or len(train_rows)+len(validation_rows)>MAX_ROWS:
        raise ValueError("Supply nonempty bounded training/validation rows.")
    if {r["house_id"] for r in train_rows} & {r["house_id"] for r in validation_rows}:
        raise ValueError("Training and validation identities must be disjoint.")
    # TODO: implement the supplied algorithm and return contract.
    raise NotImplementedError("Complete train_network before training.")



def select_model(validation_scores):
    names=("median","linear","neural")
    if set(validation_scores)!=set(names) or any(
        not number(validation_scores[name]["mae"]) or validation_scores[name]["mae"]<0
        for name in names
    ):
        raise ValueError("Supply finite nonnegative validation MAEs for all three models.")
    best=min(validation_scores[name]["mae"] for name in names)
    return next(name for name in names if validation_scores[name]["mae"]<=best+1.0)


def compare(groups, **configuration):
    train, validation, test = (groups[key] for key in ("train","validation","test"))
    ids = [row["house_id"] for rows in (train,validation,test) for row in rows]
    if len(ids)!=len(set(ids)) or not 3 <= len(ids)<=MAX_ROWS:
        raise ValueError("Use disjoint uniquely identified bounded splits.")
    scaler = fit_scaler(train)
    tx, ty = transform(train,scaler)
    linear = fit_linear(tx,ty)
    median = statistics.median(row["value_k"] for row in train)
    network = train_network(train,validation,scaler,**configuration)
    def predictions(rows):
        xs = feature_values(rows,scaler)
        return {"median":[median]*len(rows),
                "linear":restore([forward(linear,x,0)[0] for x in xs],scaler),
                "neural":restore([forward(network["parameters"],x,network["hidden_units"])[0] for x in xs],scaler)}
    val_predictions = predictions(validation)
    actual_validation = [row["value_k"] for row in validation]
    val_scores = {name:metrics(actual_validation,p) for name,p in val_predictions.items()}
    selected = select_model(val_scores)
    # No final test target is consulted before selecting the model and stopping epoch.
    test_predictions = predictions(test)
    actual_test = [row["value_k"] for row in test]
    scores = {name:{"validation":val_scores[name],"test":metrics(actual_test,p)}
              for name,p in test_predictions.items()}
    threshold = statistics.median(row["value_k"] for row in train)
    slices={}
    for label,mask in [
        *[(f"region:{region}",[row["region"]==region for row in test]) for region in sorted({r["region"] for r in test})],
        ("target:below_training_median",[y<threshold for y in actual_test]),
        ("target:at_or_above_training_median",[y>=threshold for y in actual_test])
    ]:
        indices=[i for i,include in enumerate(mask) if include]
        slices[label]={"rows":len(indices),"models":{name:metrics(
            [actual_test[i] for i in indices],[p[i] for i in indices])
            for name,p in test_predictions.items()} if indices else {}}
    return {"split_rows":{key:len(rows) for key,rows in groups.items()},
            "scaler":scaler,"linear_parameters":linear,"training_median":median,
            "neural":network,"models":scores,"selected_model":selected,
            "selection_rule":"Simplest model within 1.0 fictional value unit of the best validation MAE.",
            "target_slice_threshold":threshold,"error_slices":slices,
            "test_rows":[{"house_id":row["house_id"],"region":row["region"],"actual":row["value_k"],
                          **{name:values[i] for name,values in test_predictions.items()}}
                         for i,row in enumerate(test)]}


def plot_svg(path,title,xlabel,ylabel,series,diagonal=False,zero_line=False):
    ns="http://www.w3.org/2000/svg"
    root=ET.Element("svg",xmlns=ns,width="720",height="420",viewBox="0 0 720 420",role="img")
    ET.SubElement(root,"title").text=title
    def text(x,y,value,size=12):
        node=ET.SubElement(root,"text",x=str(x),y=str(y),fill="#172a35",
                           **{"font-family":"sans-serif","font-size":str(size)})
        node.text=str(value)
    text(60,26,title,18)
    points=[(x,y) for _,xs,ys in series for x,y in zip(xs,ys)]
    xmin,xmax=min(x for x,_ in points),max(x for x,_ in points)
    ymin,ymax=min(y for _,y in points),max(y for _,y in points)
    if diagonal:xmin=ymin=min(xmin,ymin);xmax=ymax=max(xmax,ymax)
    if zero_line:ymin=min(ymin,0);ymax=max(ymax,0)
    if xmin==xmax:xmin-=1;xmax+=1
    if ymin==ymax:ymin-=1;ymax+=1
    sx=lambda x:60+580*(x-xmin)/(xmax-xmin)
    sy=lambda y:325-265*(y-ymin)/(ymax-ymin)
    ET.SubElement(root,"path",d="M60 60V325H640",fill="none",stroke="#172a35")
    for i in range(5):
        x=xmin+(xmax-xmin)*i/4;y=ymin+(ymax-ymin)*i/4
        text(round(sx(x)-12,2),345,f"{x:.1f}");text(6,round(sy(y)+4,2),f"{y:.1f}")
    text(280,378,xlabel);text(60,48,ylabel)
    if diagonal:
        ET.SubElement(root,"line",x1=str(sx(xmin)),y1=str(sy(xmin)),x2=str(sx(xmax)),y2=str(sy(xmax)),stroke="#777",**{"stroke-dasharray":"5 4"})
        text(650,65,"y=x")
    if zero_line:
        ET.SubElement(root,"line",x1="60",y1=str(sy(0)),x2="640",y2=str(sy(0)),stroke="#777",**{"stroke-dasharray":"5 4"})
    palette=["#14746f","#c06014","#6347a3"]
    for index,(label,xs,ys) in enumerate(series):
        color=palette[index%len(palette)]
        text(60+index*195,407,label)
        ET.SubElement(root,"circle",cx=str(50+index*195),cy="403",r="3",fill=color)
        for x,y in zip(xs,ys):
            ET.SubElement(root,"circle",cx=str(sx(x)),cy=str(sy(y)),r="2.6",fill=color)
    ET.ElementTree(root).write(path,encoding="utf-8",xml_declaration=True)


def write_artifacts(report,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    (directory/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    history=report["neural"]["history"]
    with (directory/"epochs.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["epoch","train_mae","validation_mae"]);writer.writeheader();writer.writerows(history)
    with (directory/"residuals.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["house_id","region","actual","median","linear","neural"]);writer.writeheader();writer.writerows(report["test_rows"])
    plot_svg(directory/"training.svg","Neural training and validation MAE","Completed epoch","MAE (fictional value units)",[
        ("Training",[p["epoch"] for p in history],[p["train_mae"] for p in history]),
        ("Validation",[p["epoch"] for p in history],[p["validation_mae"] for p in history])])
    rows=report["test_rows"];actual=[r["actual"] for r in rows]
    plot_svg(directory/"predicted-observed.svg","Held-out predictions and observations","Observed value","Predicted value",[
        (name,actual,[r[name] for r in rows]) for name in ("median","linear","neural")],diagonal=True)
    plot_svg(directory/"residuals.svg","Held-out prediction errors","Observed value","Prediction minus observation",[
        (name,actual,[r[name]-r["actual"] for r in rows]) for name in ("median","linear","neural")],zero_line=True)
