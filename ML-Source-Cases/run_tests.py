"""Run every authored independent source contract from an isolated interpreter."""
import importlib.util
import argparse
import json
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--stdlib', action='store_true', help='Run the 208 standard-library contracts without the optional ML library graph')
args = parser.parse_args()
files = sorted(ROOT.glob('test*.py')) + sorted(ROOT.glob('ML-*/test*.py'))
if args.stdlib:
    files = [path for path in files if path.parent.name != 'ML-Legacy-Core-Contracts']
suite = unittest.TestSuite()
modules = []
for index, path in enumerate(files):
    spec = importlib.util.spec_from_file_location('source_contract_' + str(index), path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tests = unittest.defaultTestLoader.loadTestsFromModule(module)
    modules.append((path, module, tests.countTestCases()))
    suite.addTests(tests)
count = suite.countTestCases()
result = unittest.TextTestRunner(verbosity=2).run(suite)
contracts = [dict(file=path.relative_to(ROOT).as_posix(), tests=number,
                  actualProgramExecutions=getattr(module, 'EXECUTIONS', []))
             for path, module, number in modules]
print(json.dumps(dict(files=len(files), tests=count, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped), accepted=result.wasSuccessful(), scope='stdlib' if args.stdlib else 'all', contracts=contracts)))
sys.exit(0 if result.wasSuccessful() else 1)
