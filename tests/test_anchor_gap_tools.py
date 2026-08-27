import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AnchorGapToolTests(unittest.TestCase):
    def test_prepare_supports_live_dji_query_frame_names(self):
        prepare = load_script("prepare_anchor_gap_query.py")
        self.assertEqual(prepare.frame_name(4658, "query_", ".jpg"), "query_004658.jpg")

    def test_audit_supports_live_dji_query_frame_names(self):
        audit = load_script("audit_anchor_gap_recovery.py")
        self.assertEqual(audit.frame_name(3993, "query_", ".jpg"), "query_003993.jpg")


if __name__ == "__main__":
    unittest.main()
