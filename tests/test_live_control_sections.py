import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "viewer" / "index.html"
APP_PATH = ROOT / "viewer" / "app.js"
STYLE_PATH = ROOT / "viewer" / "style.css"


class LiveControlSectionTests(unittest.TestCase):
    def setUp(self):
        self.index = INDEX_PATH.read_text(encoding="utf-8")
        self.app = APP_PATH.read_text(encoding="utf-8")
        self.style = STYLE_PATH.read_text(encoding="utf-8")

    def test_all_live_control_sections_have_accessible_toggles(self):
        expected = {
            "enemy-detector": "enemy-detector-section-body",
            "flight-guard": "flight-guard-section-body",
            "localization-gate": "localization-gate-section-body",
            "mission-target": "mission-target-section-body",
            "patrol-editor": "patrol-editor-section-body",
        }
        for key, body_id in expected.items():
            with self.subTest(section=key):
                self.assertIn(f'data-live-section="{key}"', self.index)
                self.assertIn(f'aria-controls="{body_id}"', self.index)
                self.assertIn(f'id="{body_id}" class="live-control-section-body"', self.index)

    def test_live_statuses_are_outside_the_collapsible_bodies(self):
        expectations = {
            "enemy-detector-section-body": ("enemy-live-detection", "enemy-response-status"),
            "flight-guard-section-body": ("dji-command-status",),
            "localization-gate-section-body": ("localization-gate-status", "initial-position-status"),
            "mission-target-section-body": ("target-status",),
            "patrol-editor-section-body": ("patrol-status",),
        }
        for body_id, status_ids in expectations.items():
            body_start = self.index.index(f'id="{body_id}"')
            for status_id in status_ids:
                with self.subTest(body=body_id, status=status_id):
                    self.assertLess(self.index.index(f'id="{status_id}"'), body_start)

    def test_collapsing_hides_only_controls_and_persists_state(self):
        self.assertIn(".live-control-section.is-collapsed .live-control-section-body", self.style)
        self.assertNotRegex(
            self.style,
            re.compile(r"\.live-control-section\.is-collapsed\s*>?\s*\.mission-status\s*\{[^}]*display:\s*none", re.S),
        )
        self.assertIn('LIVE_CONTROL_SECTIONS_STORAGE_KEY = "atlas.liveControlCollapsedSections"', self.app)
        self.assertIn('toggle.setAttribute("aria-expanded", String(!isCollapsed))', self.app)
        self.assertIn("setupLiveControlSections();", self.app)


if __name__ == "__main__":
    unittest.main()
