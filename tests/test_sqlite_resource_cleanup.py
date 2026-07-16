from __future__ import annotations

import shutil

from app.vision.evidence import EvidenceStore
from app.vision.learned import LearnedGallery
from app.vision.outbox import VisionEventOutbox


def test_private_sqlite_stores_release_files_after_each_operation(tmp_path) -> None:
    evidence_root = tmp_path / "evidence"
    evidence = EvidenceStore(evidence_root)
    evidence.initialize()
    evidence.list_records()

    learned_root = tmp_path / "learned"
    learned = LearnedGallery(learned_root / "learned.db")
    learned.initialize()
    learned.list_references()

    outbox_root = tmp_path / "outbox"
    outbox = VisionEventOutbox(outbox_root / "outbox.db")
    outbox.initialize()
    outbox.counts()

    shutil.rmtree(evidence_root)
    shutil.rmtree(learned_root)
    shutil.rmtree(outbox_root)
