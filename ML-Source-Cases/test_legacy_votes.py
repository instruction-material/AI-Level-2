"""Execute the original classifier's actual vote tail without fitting its legacy model."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def vote_tail(source):
    tree = ast.parse(source)
    begin = next(index for index, node in enumerate(tree.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "max_label"
                         for target in node.targets))
    nodes = tree.body[begin:begin + 3]
    assert len(nodes) == 3 and isinstance(nodes[-1], ast.For)
    assert isinstance(nodes[-1].iter, ast.Name) and nodes[-1].iter.id == "k2_neighbors_count"
    return compile(ast.Module(body=nodes, type_ignores=[]), "actual-classifier-vote-tail", "exec")


class OriginalNeighborVoting(unittest.TestCase):
    def test_majority_and_existing_first_encountered_tie_behavior(self):
        source = (ROOT / "ML2-KNN-Customer-Segmentation-Classification-Updated/main.py").read_text()
        code = vote_tail(source)
        for counts, expected in [
            ({0: 2, 1: 1}, (0, 2)),
            ({0: 1, 1: 2}, (1, 2)),
            ({8: 2, 3: 2}, (8, 2)),
            ({3: 1, 8: 3, 2: 2}, (8, 3)),
        ]:
            with self.subTest(counts=counts):
                state = {"k2_neighbors_count": counts.copy()}
                exec(code, state)
                self.assertEqual((state["max_label"], state["max_count"]), expected)
                self.assertEqual(state["k2_neighbors_count"], counts)


if __name__ == "__main__":
    unittest.main()
