from __future__ import annotations

import hashlib
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.vision.evidence import (
    EvidenceCapacityError,
    EvidenceIntegrityError,
    EvidenceNotFoundError,
    EvidenceStore,
    EvidenceStoreError,
)


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def jpeg(payload: bytes) -> bytes:
    return b"\xff\xd8\xff\xe0" + payload + b"\xff\xd9"


def test_saves_with_opaque_key_and_verifies_integrity(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    store.initialize()
    scene = jpeg(b"scene")
    thumbnail = jpeg(b"thumb")

    first = store.save(scene, thumbnail=thumbnail, created_at=NOW)
    second = store.save(scene, thumbnail=thumbnail, created_at=NOW)

    assert first.reference != second.reference
    assert first.reference != hashlib.sha256(scene).hexdigest()
    assert len(first.reference) == 64
    assert store.read(first.reference) == scene
    assert store.read(first.reference, variant="thumb") == thumbnail
    assert store.verify(first.reference) is True
    assert store.path_for(first.reference).name == f"{first.reference}.jpg"
    assert store.path_for(first.reference, variant="thumb").name == (
        f"{first.reference}.thumb.jpg"
    )


def test_detects_tampering_without_returning_corrupt_data(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    store.initialize()
    record = store.save(jpeg(b"original"), created_at=NOW)
    store.path_for(record.reference).write_bytes(jpeg(b"modified"))

    with pytest.raises(EvidenceIntegrityError):
        store.read(record.reference, now=NOW)
    assert store.verify(record.reference) is False


def test_expiry_is_enforced_on_read_and_purge(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "evidence", ttl=timedelta(hours=1))
    store.initialize()
    expired = store.save(jpeg(b"old"), created_at=NOW)
    current = store.save(
        jpeg(b"current"),
        created_at=NOW + timedelta(minutes=30),
        ttl=timedelta(hours=2),
    )

    with pytest.raises(EvidenceNotFoundError, match="expirada"):
        store.read(expired.reference, now=NOW + timedelta(hours=1))
    assert store.get(expired.reference) is None

    result = store.purge(now=NOW + timedelta(hours=3))
    assert result.records == 1
    assert result.bytes_freed == current.storage_bytes
    assert store.list_records() == []


def test_capacity_rejects_new_data_without_deleting_live_evidence(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(
        tmp_path / "evidence",
        max_storage_bytes=24,
        max_item_bytes=24,
    )
    store.initialize()
    first = store.save(jpeg(b"1234567890"), created_at=NOW)

    with pytest.raises(EvidenceCapacityError, match="limite"):
        store.save(jpeg(b"abcdefghij"), created_at=NOW + timedelta(seconds=1))

    assert store.get(first.reference) is not None
    assert store.total_bytes() == first.storage_bytes
    assert len(list(store.root.glob("*.jpg"))) == 1
    assert not list(store.root.glob("*.tmp"))


def test_capacity_can_evict_oldest_when_explicitly_enabled(tmp_path: Path) -> None:
    store = EvidenceStore(
        tmp_path / "evidence",
        max_storage_bytes=24,
        max_item_bytes=24,
        evict_oldest=True,
    )
    store.initialize()
    first = store.save(jpeg(b"1234567890"), created_at=NOW)
    second = store.save(
        jpeg(b"abcdefghij"), created_at=NOW + timedelta(seconds=1)
    )

    assert store.get(first.reference) is None
    assert not store.path_for(first.reference).exists()
    assert store.get(second.reference) is not None
    assert store.total_bytes() == second.storage_bytes


def test_oversized_item_leaves_no_partial_file(tmp_path: Path) -> None:
    store = EvidenceStore(
        tmp_path / "evidence",
        max_storage_bytes=64,
        max_item_bytes=12,
    )
    store.initialize()

    with pytest.raises(EvidenceCapacityError, match="individual"):
        store.save(jpeg(b"x" * 20), created_at=NOW)

    assert store.list_records() == []
    assert not list(store.root.glob("*.jpg"))
    assert not list(store.root.glob("*.tmp"))


def test_failed_atomic_publish_leaves_no_record_or_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    store.initialize()

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated publish failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        store.save(jpeg(b"scene"), created_at=NOW)

    assert store.list_records() == []
    assert not list(store.root.glob("*.jpg"))
    assert not list(store.root.glob("*.tmp"))


def test_initialize_imports_legacy_hash_named_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    scene = jpeg(b"legacy-scene")
    thumbnail = jpeg(b"legacy-thumb")
    reference = hashlib.sha256(scene).hexdigest()
    full_path = root / f"{reference}.jpg"
    thumbnail_path = root / f"{reference}.thumb.jpg"
    full_path.write_bytes(scene)
    thumbnail_path.write_bytes(thumbnail)
    created_at = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=2)
    os.utime(full_path, (created_at.timestamp(), created_at.timestamp()))
    ttl = timedelta(days=2)

    store = EvidenceStore(root, ttl=ttl)
    store.initialize()

    record = store.get(reference)
    assert record is not None
    assert record.reference == reference
    assert record.scene_sha256 == reference
    assert record.thumbnail_sha256 == hashlib.sha256(thumbnail).hexdigest()
    assert record.created_at == created_at
    assert record.expires_at == created_at + ttl
    assert store.read(reference) == scene
    assert store.read(reference, variant="thumb") == thumbnail

    store.initialize()
    assert store.verify(reference) is True
    assert len(store.list_records()) == 1
    assert full_path.exists()
    assert thumbnail_path.exists()


def test_store_rejects_a_conflicting_retention_policy(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    EvidenceStore(root, ttl=timedelta(days=30), evict_oldest=True).initialize()

    with pytest.raises(EvidenceStoreError, match="política"):
        EvidenceStore(
            root,
            ttl=timedelta(days=7),
            evict_oldest=True,
        ).initialize()


def test_legacy_import_applies_ttl_and_storage_quota(tmp_path: Path) -> None:
    current = datetime.now(UTC).replace(microsecond=0)
    expired_root = tmp_path / "expired"
    expired_root.mkdir()
    expired_scene = jpeg(b"expired")
    expired_reference = hashlib.sha256(expired_scene).hexdigest()
    expired_path = expired_root / f"{expired_reference}.jpg"
    expired_path.write_bytes(expired_scene)
    expired_time = current - timedelta(hours=2)
    os.utime(expired_path, (expired_time.timestamp(), expired_time.timestamp()))

    expired_store = EvidenceStore(expired_root, ttl=timedelta(hours=1))
    expired_store.initialize()

    assert expired_store.get(expired_reference) is None
    assert not expired_path.exists()

    quota_root = tmp_path / "quota"
    quota_root.mkdir()
    older_scene = jpeg(b"legacy-old")
    newer_scene = jpeg(b"legacy-new")
    assert len(older_scene) == len(newer_scene)
    older_reference = hashlib.sha256(older_scene).hexdigest()
    newer_reference = hashlib.sha256(newer_scene).hexdigest()
    older_path = quota_root / f"{older_reference}.jpg"
    newer_path = quota_root / f"{newer_reference}.jpg"
    older_path.write_bytes(older_scene)
    newer_path.write_bytes(newer_scene)
    older_time = current - timedelta(minutes=20)
    newer_time = current - timedelta(minutes=10)
    os.utime(older_path, (older_time.timestamp(), older_time.timestamp()))
    os.utime(newer_path, (newer_time.timestamp(), newer_time.timestamp()))

    quota_store = EvidenceStore(
        quota_root,
        max_storage_bytes=len(newer_scene),
        max_item_bytes=len(newer_scene),
    )
    quota_store.initialize()

    assert quota_store.get(newer_reference) is not None
    assert newer_path.exists()
    assert quota_store.get(older_reference) is None
    assert not older_path.exists()
    assert quota_store.total_bytes() == len(newer_scene)


def test_initialize_recovers_orphan_left_by_process_crash(
    tmp_path: Path,
) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    class CrashAfterPublishStore(EvidenceStore):
        def _write_atomic(self, destination: Path, data: bytes) -> None:
            super()._write_atomic(destination, data)
            raise SimulatedProcessCrash

    root = tmp_path / "evidence"
    store = EvidenceStore(root)
    store.initialize()
    intact = store.save(
        jpeg(b"intact-scene"),
        thumbnail=jpeg(b"intact-thumb"),
        created_at=datetime.now(UTC),
    )

    crashing_store = CrashAfterPublishStore(root)
    crashing_store.initialize()
    with pytest.raises(SimulatedProcessCrash):
        crashing_store.save(jpeg(b"orphan"), created_at=datetime.now(UTC))

    assert len(list(root.glob("*.jpg"))) == 3

    recovered = EvidenceStore(root)
    recovered.initialize()

    assert recovered.read(intact.reference) == jpeg(b"intact-scene")
    assert recovered.read(intact.reference, variant="thumb") == jpeg(b"intact-thumb")
    assert {path.name for path in root.glob("*.jpg")} == {
        f"{intact.reference}.jpg",
        f"{intact.reference}.thumb.jpg",
    }


def test_initialize_removes_old_managed_temporary_on_all_platforms(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    store.initialize()
    old_temporary = store.root / f".{'a' * 64}.jpg.{'1' * 32}.tmp"
    old_legacy_temporary = store.root / f"{'f' * 64}.thumb.jpg.tmp"
    recent_temporary = store.root / f".{'b' * 64}.thumb.jpg.{'2' * 32}.tmp"
    unrelated_temporary = store.root / ".manual-preview.tmp"
    old_temporary.write_bytes(b"old")
    old_legacy_temporary.write_bytes(b"old-legacy")
    recent_temporary.write_bytes(b"recent")
    unrelated_temporary.write_bytes(b"unrelated")
    current = datetime.now(UTC)
    old_timestamp = (current - timedelta(hours=2)).timestamp()
    os.utime(old_temporary, (old_timestamp, old_timestamp))
    os.utime(old_legacy_temporary, (old_timestamp, old_timestamp))

    store.initialize()

    assert not old_temporary.exists()
    assert not old_legacy_temporary.exists()
    assert recent_temporary.exists()
    assert unrelated_temporary.exists()


def test_purge_reconciles_orphans_without_removing_live_evidence(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    store.initialize()
    intact = store.save(
        jpeg(b"intact-scene"),
        thumbnail=jpeg(b"intact-thumb"),
        created_at=NOW,
        ttl=timedelta(hours=4),
    )
    orphan_reference = "c" * 64
    orphan_full = store.path_for(orphan_reference)
    orphan_thumbnail = store.path_for(orphan_reference, variant="thumb")
    orphan_full.write_bytes(jpeg(b"orphan-scene"))
    orphan_thumbnail.write_bytes(jpeg(b"orphan-thumb"))
    old_temporary = store.root / f".{'d' * 64}.jpg.{'3' * 32}.tmp"
    recent_temporary = store.root / f".{'e' * 64}.jpg.{'4' * 32}.tmp"
    old_temporary.write_bytes(b"old")
    recent_temporary.write_bytes(b"recent")
    os.utime(old_temporary, (NOW.timestamp(), NOW.timestamp()))
    recent_time = NOW + timedelta(hours=1, minutes=30)
    os.utime(recent_temporary, (recent_time.timestamp(), recent_time.timestamp()))
    unrelated_jpeg = store.root / "manual.jpg"
    unrelated_jpeg.write_bytes(jpeg(b"manual"))

    result = store.purge(now=NOW + timedelta(hours=2))

    assert result.records == 0
    assert not orphan_full.exists()
    assert not orphan_thumbnail.exists()
    assert not old_temporary.exists()
    assert recent_temporary.exists()
    assert unrelated_jpeg.exists()
    assert store.read(intact.reference, now=NOW + timedelta(hours=2)) == jpeg(
        b"intact-scene"
    )
    assert store.read(
        intact.reference,
        variant="thumb",
        now=NOW + timedelta(hours=2),
    ) == jpeg(b"intact-thumb")


def test_reconciliation_unlinks_managed_symlink_without_following_it(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    store.initialize()
    record = store.save(jpeg(b"original"), created_at=NOW)
    outside = tmp_path / "outside.jpg"
    outside_data = jpeg(b"outside")
    outside.write_bytes(outside_data)
    link = store.path_for(record.reference)
    link.unlink()
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("O ambiente não permite criar symlink.")

    with pytest.raises(EvidenceIntegrityError, match="regular"):
        store.read(record.reference, now=NOW)
    store.purge(now=NOW)

    assert not os.path.lexists(link)
    assert store.get(record.reference) is not None
    assert outside.read_bytes() == outside_data


def test_legacy_import_does_not_follow_hash_named_symlink(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside_data = jpeg(b"legacy-outside")
    outside.write_bytes(outside_data)
    reference = hashlib.sha256(outside_data).hexdigest()
    link = root / f"{reference}.jpg"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("O ambiente não permite criar symlink.")

    store = EvidenceStore(root)
    store.initialize()

    assert store.get(reference) is None
    assert not os.path.lexists(link)
    assert outside.read_bytes() == outside_data


def test_delete_is_idempotent_and_reference_cannot_escape_root(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    store.initialize()
    record = store.save(jpeg(b"scene"), created_at=NOW)

    with pytest.raises(ValueError, match="Referência"):
        store.path_for("../../outside")
    assert store.delete(record.reference) is True
    assert store.delete(record.reference) is False
    assert not store.path_for(record.reference).exists()


def test_private_permissions_are_applied_on_posix(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("ACLs do Windows são herdadas do diretório privado.")
    store = EvidenceStore(tmp_path / "evidence")
    store.initialize()
    record = store.save(jpeg(b"scene"), created_at=NOW)

    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.database_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path_for(record.reference).stat().st_mode) == 0o600


def test_rejects_a_link_as_storage_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "evidence"
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("O ambiente não permite criar symlink.")

    with pytest.raises(EvidenceStoreError, match="link"):
        EvidenceStore(link).initialize()
