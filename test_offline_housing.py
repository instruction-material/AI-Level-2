import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
FOLDERS = ["ML6-Offline-Housing-Comparison",
           "ML-11-ml6-regression-with-neural-networks-supplemental-2",
           "ML-12-ml6-regression-with-neural-networks-supplemental-3"]


def load(folder=FOLDERS[0], role="solution"):
    path = ROOT / folder / role / "model.py"
    spec = importlib.util.spec_from_file_location("housing_"+folder+role, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(model):
    return model.load_rows(ROOT / FOLDERS[0] / "solution/data/synthetic_housing.csv")


def run(script, *arguments):
    with tempfile.TemporaryDirectory() as elsewhere:
        result = subprocess.run([sys.executable, "-B", str(script), *map(str, arguments)],
                                cwd=elsewhere, capture_output=True, text=True,
                                check=True, timeout=15)
    return json.loads(result.stdout)


class OfflineHousingContracts(unittest.TestCase):
    def setUp(self):
        self.model = load()
        self.groups = fixture(self.model)

    def test_three_fixture_identities_and_counts(self):
        root = ROOT / "ML-Offline-Case-Fixtures"
        manifest = json.loads((root / "manifest.json").read_text())
        for filename, count in [("garden_growth.csv",120),("support_tickets.csv",240),("synthetic_housing.csv",120)]:
            with self.subTest(filename=filename):
                self.assertEqual(hashlib.sha256((root/filename).read_bytes()).hexdigest(), manifest["files"][filename]["sha256"])
                with (root/filename).open(newline="") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), count)
                self.assertEqual(list(rows[0]), manifest["files"][filename]["columns"])

    def test_ticket_class_counts_and_group_disjointness_contract(self):
        with (ROOT/"ML-Offline-Case-Fixtures/support_tickets.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual({label:sum(row["label"]==label for row in rows) for label in ("question","bug","request")},
                         {"question":120,"bug":80,"request":40})
        for family in {row["template_family"] for row in rows}:
            self.assertEqual(len({row["label"] for row in rows if row["template_family"]==family}),1)

    def test_fixture_generator_recreates_exact_bytes_elsewhere(self):
        source = ROOT/"ML-Offline-Case-Fixtures"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copy2(source/"generate.py",root/"generate.py")
            result = subprocess.run([sys.executable,"-I","-B",str(root/"generate.py")],
                                    cwd="/",capture_output=True,check=True,timeout=15)
            self.assertTrue(result.stdout)
            for name in ["garden_growth.csv","support_tickets.csv","synthetic_housing.csv","manifest.json"]:
                self.assertEqual((root/name).read_bytes(),(source/name).read_bytes())

    def test_fixed_splits_are_complete_and_disjoint(self):
        self.assertEqual({k:len(v) for k,v in self.groups.items()},{"train":72,"validation":24,"test":24})
        ids = [row["house_id"] for group in self.groups.values() for row in group]
        self.assertEqual(len(ids),len(set(ids)))
        for rows in self.groups.values():
            self.assertEqual({row["region"] for row in rows},{"North","South","West"})

    def test_each_role_contains_identical_required_fixture(self):
        expected = (ROOT/"ML-Offline-Case-Fixtures/synthetic_housing.csv").read_bytes()
        for folder in FOLDERS:
            for role in ("starter","solution"):
                self.assertEqual((ROOT/folder/role/"data/synthetic_housing.csv").read_bytes(),expected)

    def test_scaler_uses_only_training_rows(self):
        scaler = self.model.fit_scaler(self.groups["train"])
        expected = [statistics.fmean(row[key] for row in self.groups["train"])
                    for key in ("area_m2","rooms","value_k")]
        self.assertEqual(scaler["means"],expected)
        self.assertEqual(scaler["fit_rows"],72)
        before = copy.deepcopy(scaler)
        changed = copy.deepcopy(self.groups["test"])
        for row in changed:
            row["area_m2"] += 1e6
            row["value_k"] += 1e7
        self.model.transform(changed,scaler)
        self.assertEqual(scaler,before)

    def test_constant_columns_use_scale_one(self):
        rows = [{"area_m2":2,"rooms":3,"value_k":7}]*3
        scaler = self.model.fit_scaler(rows)
        self.assertEqual(scaler["scales"],[1,1,1])
        self.assertEqual(self.model.transform(rows,scaler),([[0,0]]*3,[0]*3))

    def test_prediction_features_do_not_require_or_read_targets(self):
        class Query(dict):
            def __getitem__(self,key):
                if key=="value_k":
                    raise AssertionError("A prediction must not read its target.")
                return super().__getitem__(key)
        scaler = self.model.fit_scaler(self.groups["train"])
        values = self.model.feature_values([Query(area_m2=100,rooms=2)],scaler)
        self.assertEqual(len(values),1)
        self.assertEqual(len(values[0]),2)

    def test_metrics_against_hand_calculation(self):
        actual = self.model.metrics([10,20],[12,17])
        self.assertEqual(actual["mae"],2.5)
        self.assertAlmostEqual(actual["rmse"],math.sqrt(6.5))
        for xs,ys in [([],[]),([1],[]),([1],[math.inf])]:
            with self.subTest(xs=xs),self.assertRaises(ValueError):
                self.model.metrics(xs,ys)

    def test_exact_linear_baseline_on_known_plane(self):
        xs = [[0,0],[1,0],[0,1],[1,1]]
        ys = [1,3,4,6]
        coefficients = self.model.fit_linear(xs,ys)
        for a,b in zip(coefficients,[2,3,1]):
            self.assertAlmostEqual(a,b)
        self.assertAlmostEqual(self.model.forward(coefficients,[0.5,0.25],0)[0],2.75)

    def test_regression_output_and_derivative_are_unbounded_linear(self):
        loss,gradient = self.model.loss_gradient([0,0,0],[[0,1]],[2],0)
        self.assertEqual((loss,gradient),(4,[0,-4,-4]))
        self.assertEqual(self.model.forward([0,0,5],[0,1],0)[0],5)

    def test_all_linear_and_hidden_gradients_against_finite_differences(self):
        for hidden in [0,3,8]:
            count = 3 if hidden==0 else 4*hidden+1
            parameters = [-0.1+0.01*i for i in range(count)]
            xs,ys = [[0.2,-0.4],[-0.3,0.5]],[0.7,-0.2]
            _,gradient = self.model.loss_gradient(parameters,xs,ys,hidden,l2=0.01)
            for index in range(count):
                high,low = parameters[:],parameters[:]
                high[index]+=1e-6;low[index]-=1e-6
                fhigh,_ = self.model.loss_gradient(high,xs,ys,hidden,l2=0.01)
                flow,_ = self.model.loss_gradient(low,xs,ys,hidden,l2=0.01)
                with self.subTest(hidden=hidden,parameter=index):
                    self.assertAlmostEqual(gradient[index],(fhigh-flow)/2e-6,delta=2e-9)

    def test_l2_regularizes_weights_and_preserves_biases(self):
        p = [0.1]*13
        loss0,g0 = self.model.loss_gradient(p,[[0,1]],[0.2],3,l2=0)
        loss1,g1 = self.model.loss_gradient(p,[[0,1]],[0.2],3,l2=0.01)
        weights={0,1,3,4,6,7,9,10,11}
        self.assertAlmostEqual(loss1-loss0,0.01*9*0.1**2)
        for i in range(13):
            self.assertAlmostEqual(g1[i]-g0[i],0.002 if i in weights else 0)

    def test_best_validation_parameters_and_epoch_are_returned(self):
        train,val = self.groups["train"],self.groups["validation"]
        scaler = self.model.fit_scaler(train)
        for hidden in [0,3]:
            state = self.model.train_network(train,val,scaler,hidden_units=hidden)
            history = state["history"]
            best = history[state["best_epoch"]]
            self.assertAlmostEqual(best["validation_mae"],min(x["validation_mae"] for x in history),delta=1e-6)
            xs,_ = self.model.transform(val,scaler)
            predictions = self.model.restore([self.model.forward(state["parameters"],x,hidden)[0] for x in xs],scaler)
            measured = self.model.metrics([r["value_k"] for r in val],predictions)["mae"]
            self.assertAlmostEqual(measured,best["validation_mae"],places=10)
            self.assertLessEqual(state["completed_epochs"],30)
            if hidden==0:
                self.assertEqual((state["best_epoch"],state["completed_epochs"],state["stop_reason"]),
                                 (8,14,"validation_patience"))

    def test_cpu_budget_returns_a_valid_saved_checkpoint(self):
        scaler = self.model.fit_scaler(self.groups["train"])
        with patch.object(self.model.time,"process_time",side_effect=[0,100,100]):
            state = self.model.train_network(self.groups["train"],self.groups["validation"],scaler,cpu_seconds=0.1)
        self.assertEqual((state["completed_epochs"],state["best_epoch"],state["stop_reason"]),(0,0,"cpu_budget"))
        self.assertEqual(len(state["history"]),1)

    def test_training_caps_and_disjointness_are_enforced(self):
        train,val = self.groups["train"],self.groups["validation"]
        scaler = self.model.fit_scaler(train)
        for config in [{"epochs":31},{"batch_size":33},{"hidden_units":9},{"cpu_seconds":6},
                       {"learning_rate":0},{"l2":0.2},{"patience":0},{"epochs":True}]:
            with self.subTest(config=config),self.assertRaises(ValueError):
                self.model.train_network(train,val,scaler,**config)
        with self.assertRaises(ValueError):
            self.model.train_network(train,train,scaler)

    def test_seeded_training_preserves_global_random_and_inputs(self):
        train,val = self.groups["train"],self.groups["validation"]
        before = copy.deepcopy(self.groups);state = random.getstate()
        scaler = self.model.fit_scaler(train)
        first = self.model.train_network(train,val,scaler,epochs=3)
        second = self.model.train_network(train,val,scaler,epochs=3)
        first.pop("cpu_seconds");second.pop("cpu_seconds")
        self.assertEqual(first,second)
        self.assertEqual(self.groups,before)
        self.assertEqual(random.getstate(),state)

    def test_test_targets_cannot_change_fitting_selection_or_stopping(self):
        first = self.model.compare(self.groups)
        changed = copy.deepcopy(self.groups)
        for row in changed["test"]:
            row["value_k"]+=1000
        second = self.model.compare(changed)
        for key in ["scaler","linear_parameters","training_median","selected_model","target_slice_threshold"]:
            self.assertEqual(first[key],second[key])
        for key in ["parameters","best_epoch","completed_epochs","history","stop_reason"]:
            self.assertEqual(first["neural"][key],second["neural"][key])
        self.assertNotEqual(first["models"]["neural"]["test"],second["models"]["neural"]["test"])
        for key in ["median","linear","neural"]:
            self.assertEqual(first["models"][key]["validation"],second["models"][key]["validation"])

    def test_error_slices_partition_test_rows(self):
        report = self.model.compare(self.groups)
        self.assertEqual(sum(v["rows"] for k,v in report["error_slices"].items() if k.startswith("region:")),24)
        self.assertEqual(sum(v["rows"] for k,v in report["error_slices"].items() if k.startswith("target:")),24)
        self.assertEqual(len(report["test_rows"]),24)

    def test_supplied_comparison_result_prefers_simpler_model(self):
        report = self.model.compare(self.groups)
        self.assertEqual(report["selected_model"],"linear")
        self.assertAlmostEqual(report["models"]["linear"]["test"]["mae"],5.6768609266,places=8)
        self.assertAlmostEqual(report["models"]["median"]["test"]["mae"],38.8248333333,places=8)
        self.assertAlmostEqual(report["models"]["neural"]["test"]["mae"],7.6212524987,places=8)

    def test_validation_simplicity_rule_allows_each_candidate_to_win(self):
        for values,expected in [([10.5,10,9.8],"median"),([20,11,10],"linear"),([20,12,10],"neural")]:
            scores={name:{"mae":value} for name,value in zip(["median","linear","neural"],values)}
            self.assertEqual(self.model.select_model(scores),expected)

    def test_starters_have_real_unfinished_assignment_functions(self):
        for folder in FOLDERS:
            model = load(folder,"starter")
            groups = model.load_rows(ROOT/folder/"starter/data/synthetic_housing.csv")
            with self.subTest(folder=folder),self.assertRaises(NotImplementedError):
                model.compare(groups)

    def test_source_roles_run_from_another_directory(self):
        for folder in FOLDERS:
            report=run(ROOT/folder/"solution/main.py")
            self.assertEqual(report["split_rows"],{"train":72,"validation":24,"test":24})
            self.assertEqual(report["selected_model"],"linear")
            self.assertLessEqual(report["completed_epochs"],30)

    def test_exported_reference_generates_complete_reports_and_plots(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/"results"
            run(ROOT/FOLDERS[0]/"solution/main.py","--output",directory)
            report=json.loads((directory/"report.json").read_text())
            self.assertEqual(len(report["test_rows"]),24)
            with (directory/"residuals.csv").open(newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))),24)
            with (directory/"epochs.csv").open(newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))),31)
            ns={"s":"http://www.w3.org/2000/svg"}
            for name in ["training.svg","predicted-observed.svg","residuals.svg"]:
                root=ET.parse(directory/name).getroot()
                self.assertIsNotNone(root.find("s:title",ns))
                self.assertTrue(root.findall("s:circle",ns))
            text=(directory/"predicted-observed.svg").read_text()
            self.assertIn("y=x",text)

    def test_actual_exported_cli_never_uses_test_labels_to_select(self):
        before=run(ROOT/FOLDERS[0]/"solution/main.py")
        with tempfile.TemporaryDirectory() as temp:
            role=Path(temp)/"role";shutil.copytree(ROOT/FOLDERS[0]/"solution",role)
            data=role/"data/synthetic_housing.csv"
            with data.open(newline="") as f:
                reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
            for row in rows:
                if row["split"]=="test":row["value_k"]=float(row["value_k"])+1000
            with data.open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            after=run(role/"main.py")
        for key in ["selected_model","best_epoch","completed_epochs","stop_reason"]:
            self.assertEqual(before[key],after[key])
        for name in ["median","linear","neural"]:
            self.assertEqual(before["models"][name]["validation"],after["models"][name]["validation"])
        self.assertNotEqual(before["dataset_sha256"],after["dataset_sha256"])

    def test_intro_data_card_has_actual_rows_and_stable_restart(self):
        script=ROOT/"ML0-Offline-Data-Card/solution/main.py"
        first,second=run(script),run(script)
        self.assertEqual(first,second)
        self.assertEqual((first["rows"],first["soil_counts"]),(120,{"clay":40,"loam":40,"sand":40}))
        self.assertTrue(all(count==0 for count in first["missing"].values()))
        self.assertEqual(set(first["data_card"]["features"]),{"sun_hours","water_liters","soil_type"})
        self.assertIn("python",first["environment"])
        self.assertEqual(set(first["environment"]["packages"]),{"scikit-learn","pandas","numpy","matplotlib","jupyter"})

    def test_intro_raw_split_is_complete_and_reproducible(self):
        report=run(ROOT/"ML0-Offline-Data-Card/solution/main.py")
        split=report["split"]
        self.assertEqual((len(split["train_ids"]),len(split["held_out_ids"])),(96,24))
        self.assertFalse(set(split["train_ids"])&set(split["held_out_ids"]))
        self.assertEqual(len(set(split["train_ids"]+split["held_out_ids"])),120)

    def test_evaluation_contract_matches_the_supplied_fixture(self):
        contract=run(ROOT/"ML7-5-Offline-Evaluation-Contract/solution/main.py")
        self.assertEqual(contract["class_counts"],{"question":120,"bug":80,"request":40})
        self.assertEqual(contract["split_rows"],{"train":132,"validation":54,"test":54})
        self.assertEqual(contract["prediction_input"],"summary")
        self.assertEqual((contract["primary_metric"],contract["secondary_metric"]),("macro_f1","balanced_accuracy"))
        for name in ["train","validation","test"]:
            self.assertEqual(set(contract["split_class_counts"][name]),{"bug","question","request"})
        self.assertIn("training summaries",contract["vectorization"])
        self.assertIn("Freeze",contract["test_gate"])

    def test_evaluation_contract_families_and_rows_never_overlap(self):
        contract=run(ROOT/"ML7-5-Offline-Evaluation-Contract/solution/main.py")
        for key in ["split_ids","split_families"]:
            parts=contract[key]
            self.assertFalse(set(parts["train"])&set(parts["validation"]))
            self.assertFalse(set(parts["train"])&set(parts["test"]))
            self.assertFalse(set(parts["validation"])&set(parts["test"]))
        self.assertEqual(len(set(sum(contract["split_ids"].values(),[]))),240)

    def test_evaluation_contract_seed_repeat_and_variant(self):
        script=ROOT/"ML7-5-Offline-Evaluation-Contract/solution/main.py"
        first,repeat,other=run(script),run(script),run(script,"--seed",43)
        self.assertEqual(first,repeat)
        self.assertNotEqual(first["split_ids"],other["split_ids"])
        self.assertEqual(first["split_class_counts"],other["split_class_counts"])

    def test_evaluation_export_is_complete_machine_readable_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"nested/contract.json"
            stdout=run(ROOT/"ML7-5-Offline-Evaluation-Contract/solution/main.py","--output",path)
            self.assertEqual(json.loads(path.read_text()),stdout)

    def test_intro_starters_are_explicitly_unfinished(self):
        for folder in ["ML0-Offline-Data-Card","ML7-5-Offline-Evaluation-Contract"]:
            script=ROOT/folder/"starter/main.py"
            with self.subTest(folder=folder),tempfile.TemporaryDirectory() as temp:
                result=subprocess.run([sys.executable,"-B",str(script)],cwd=temp,capture_output=True,text=True,timeout=15)
                self.assertNotEqual(result.returncode,0)
                self.assertIn("NotImplementedError",result.stderr)

    def test_intro_roles_embed_their_exact_owned_input_files(self):
        for folder,name in [("ML0-Offline-Data-Card","garden_growth.csv"),
                            ("ML7-5-Offline-Evaluation-Contract","support_tickets.csv")]:
            for role in ["starter","solution"]:
                self.assertEqual((ROOT/folder/role/"data"/name).read_bytes(),
                                 (ROOT/"ML-Offline-Case-Fixtures"/name).read_bytes())

    def test_invalid_csv_duplicate_identity_and_missing_split(self):
        original=ROOT/FOLDERS[0]/"solution/data/synthetic_housing.csv"
        with original.open(newline="") as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;source=list(reader)
        for mode in ["duplicate","missing_split","nonfinite","row_cap"]:
            rows=copy.deepcopy(source)
            if mode=="duplicate":rows[1]["house_id"]=rows[0]["house_id"]
            elif mode=="missing_split":
                for row in rows:
                    if row["split"]=="test":row["split"]="validation"
            elif mode=="nonfinite":rows[0]["area_m2"]="nan"
            else:
                rows=[{**source[i%len(source)],"house_id":f"over-{i}"} for i in range(501)]
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/"data.csv"
                with path.open("w",newline="") as f:
                    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
                with self.subTest(mode=mode),self.assertRaises(ValueError):
                    self.model.load_rows(path)


if __name__=="__main__":
    unittest.main()
