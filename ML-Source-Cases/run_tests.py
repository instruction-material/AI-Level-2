"""Run every authored independent source contract from an isolated interpreter."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parent.parent
files = sorted(ROOT.glob('test*.py')) + sorted(ROOT.glob('ML-*/test*.py'))
suite = unittest.TestSuite()
for index, path in enumerate(files):
    spec = importlib.util.spec_from_file_location('source_contract_' + str(index), path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
count = suite.countTestCases()
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps(dict(files=len(files), tests=count, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped), accepted=result.wasSuccessful())))
sys.exit(0 if result.wasSuccessful() else 1)
