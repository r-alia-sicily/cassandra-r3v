from __future__ import annotations

import json
import unittest
from pathlib import Path

from cassandra_r3v import __version__
from cassandra_r3v.cli import build_parser
from cassandra_r3v.config import AutoStructureConfig, TrainingConfig


ROOT = Path(__file__).resolve().parents[1]


class ReleaseContractTests(unittest.TestCase):
    @classmethod
    def profile(cls) -> dict:
        return json.loads((ROOT / "examples" / "article_profile.json").read_text(encoding="utf-8"))

    def test_article_profile_matches_code_defaults(self):
        profile = self.profile()
        self.assertEqual(profile["release"], __version__)
        self.assertEqual(
            TrainingConfig.from_dict(profile["training"]).to_dict(),
            TrainingConfig().validate().to_dict(),
        )
        self.assertEqual(
            AutoStructureConfig.from_dict(profile["autostructure"]).to_dict(),
            AutoStructureConfig().validate().to_dict(),
        )

    def test_headless_autostructure_is_exposed(self):
        args = build_parser().parse_args(["autostructure"])
        self.assertEqual(args.command, "autostructure")
        self.assertFalse(args.resume_current)
        self.assertEqual(args.seeds, [7, 42, 99])
        self.assertEqual(args.max_candidates, 10_000)

    def test_public_material_has_one_release_identity(self):
        public_paths = [
            ROOT / "README.md",
            ROOT / "CHANGELOG.md",
            ROOT / "CITATION.cff",
            ROOT / "docs",
            ROOT / "examples",
            ROOT / "src" / "cassandra_r3v" / "resources",
        ]
        files = []
        for path in public_paths:
            files.extend(path.rglob("*") if path.is_dir() else [path])
        text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace").lower()
            for path in files
            if path.is_file()
        )
        forbidden = ["v" + number for number in ("15", "17", "18")]
        forbidden.extend(("development " + "build", "frozen " + "manuscript"))
        for phrase in forbidden:
            self.assertNotIn(phrase, text)

    def test_packaged_method_document_matches_repository(self):
        public = (ROOT / "docs" / "METHOD_ALIGNMENT.md").read_text(encoding="utf-8")
        packaged = (
            ROOT / "src" / "cassandra_r3v" / "resources" / "METHOD_ALIGNMENT.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(public, packaged)

    def test_github_publication_files_are_present(self):
        required = (
            ".github/workflows/ci.yml",
            ".github/ISSUE_TEMPLATE/bug_report.yml",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "docs/REPRODUCIBILITY.md",
            "docs/PUBLISHING.md",
            "docs/RELEASE_NOTES.md",
            "docs/Cassandra_User_Manual_v1.0.0.tex",
            "docs/Cassandra_User_Manual_v1.0.0.pdf",
            "docs/assets/cassandra_gui_autostructure.png",
        )
        self.assertTrue(all((ROOT / name).is_file() for name in required))


if __name__ == "__main__":
    unittest.main()
