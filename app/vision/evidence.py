from __future__ import annotations

import hashlib
import os
import re
import secrets
import sqlite3
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal


class EvidenceStoreError(RuntimeError):
    pass


class EvidenceNotFoundError(EvidenceStoreError):
    pass


class EvidenceIntegrityError(EvidenceStoreError):
    pass


class EvidenceCapacityError(EvidenceStoreError):
    pass


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    reference: str
    scene_sha256: str
    scene_bytes: int
    thumbnail_sha256: str | None
    thumbnail_bytes: int
    media_type: str
    created_at: datetime
    expires_at: datetime

    @property
    def storage_bytes(self) -> int:
        return self.scene_bytes + self.thumbnail_bytes


@dataclass(frozen=True, slots=True)
class PurgeResult:
    records: int
    bytes_freed: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_records (
    reference TEXT PRIMARY KEY,
    scene_sha256 TEXT NOT NULL,
    scene_bytes INTEGER NOT NULL,
    thumbnail_sha256 TEXT,
    thumbnail_bytes INTEGER NOT NULL DEFAULT 0,
    media_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    CHECK (length(reference) = 64),
    CHECK (length(scene_sha256) = 64),
    CHECK (scene_bytes > 0),
    CHECK (thumbnail_sha256 IS NULL OR length(thumbnail_sha256) = 64),
    CHECK (thumbnail_bytes >= 0)
)
"""
_POLICY_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_store_policy (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    ttl_microseconds INTEGER NOT NULL,
    max_storage_bytes INTEGER NOT NULL,
    max_item_bytes INTEGER NOT NULL,
    evict_oldest INTEGER NOT NULL CHECK (evict_oldest IN (0, 1))
)
"""

_HEX = frozenset("0123456789abcdef")
_TEMPORARY_MAX_AGE = timedelta(hours=1)
_MAX_MTIME_SKEW = timedelta(minutes=5)
_EVIDENCE_FILE_PATTERN = re.compile(
    r"^(?P<reference>[0-9a-f]{64})(?P<thumbnail>\.thumb)?\.jpg$",
    re.IGNORECASE,
)
_TEMPORARY_FILE_PATTERN = re.compile(
    r"^(?:\.(?P<destination>[0-9a-f]{64}(?:\.thumb)?\.jpg)"
    r"\.(?P<nonce>[0-9a-f]{32})|"
    r"(?P<legacy>[0-9a-f]{64}(?:\.thumb)?\.jpg))\.tmp$",
    re.IGNORECASE,
)


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("O instante precisa incluir fuso horário.")
    return value.astimezone(UTC).isoformat()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvidenceStoreError("O índice de evidências possui data inválida.")
    return parsed.astimezone(UTC)


class EvidenceStore:
    def __init__(
        self,
        root: str | Path,
        *,
        ttl: timedelta = timedelta(days=30),
        max_storage_bytes: int = 10 * 1024 * 1024 * 1024,
        max_item_bytes: int = 25 * 1024 * 1024,
        evict_oldest: bool = False,
    ) -> None:
        if ttl <= timedelta(0):
            raise ValueError("ttl precisa ser positivo.")
        if max_storage_bytes < 1 or max_item_bytes < 1:
            raise ValueError("Os limites de armazenamento precisam ser positivos.")
        if max_item_bytes > max_storage_bytes:
            raise ValueError("max_item_bytes não pode superar max_storage_bytes.")
        self.root = Path(root)
        self.ttl = ttl
        self.max_storage_bytes = max_storage_bytes
        self.max_item_bytes = max_item_bytes
        self.evict_oldest = evict_oldest
        self._policy = (
            round(ttl.total_seconds() * 1_000_000),
            max_storage_bytes,
            max_item_bytes,
            int(evict_oldest),
        )
        self.database_path = self.root / ".evidence-index.sqlite3"

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        if os.path.lexists(self.root) and self._is_link_or_reparse(self.root):
            raise EvidenceStoreError("O diretório de evidências não pode ser um link.")
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise EvidenceStoreError("O caminho de evidências não é um diretório.")
        self._set_private_permissions(self.root, directory=True)
        if os.path.lexists(self.database_path) and self._is_link_or_reparse(
            self.database_path
        ):
            raise EvidenceStoreError("O índice de evidências não pode ser um link.")
        with self._connect() as connection:
            connection.execute(_SCHEMA)
            connection.execute(_POLICY_SCHEMA)
            policy = connection.execute(
                """
                SELECT ttl_microseconds, max_storage_bytes,
                       max_item_bytes, evict_oldest
                FROM evidence_store_policy
                WHERE singleton=1
                """
            ).fetchone()
            if policy is None:
                connection.execute(
                    """
                    INSERT INTO evidence_store_policy (
                        singleton, ttl_microseconds, max_storage_bytes,
                        max_item_bytes, evict_oldest
                    ) VALUES (1, ?, ?, ?, ?)
                    """,
                    self._policy,
                )
            elif tuple(int(value) for value in policy) != self._policy:
                raise EvidenceStoreError(
                    "A política de evidências difere da política gravada no índice."
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_evidence_expiry
                ON evidence_records(expires_at, created_at)
                """
            )
        self._set_private_permissions(self.database_path, directory=False)
        self._reconcile_files(now=datetime.now(UTC))

    def save(
        self,
        scene: bytes | bytearray | memoryview,
        *,
        thumbnail: bytes | bytearray | memoryview | None = None,
        created_at: datetime | None = None,
        ttl: timedelta | None = None,
        media_type: str = "image/jpeg",
    ) -> EvidenceRecord:
        if media_type != "image/jpeg":
            raise ValueError("Somente evidências JPEG são aceitas.")
        scene_data = bytes(scene)
        thumbnail_data = bytes(thumbnail) if thumbnail is not None else None
        self._validate_jpeg(scene_data, name="cena")
        if thumbnail_data is not None:
            self._validate_jpeg(thumbnail_data, name="miniatura")
        total_size = len(scene_data) + (
            len(thumbnail_data) if thumbnail_data is not None else 0
        )
        if total_size > self.max_item_bytes:
            raise EvidenceCapacityError("A evidência excede o limite individual.")
        effective_ttl = ttl or self.ttl
        if effective_ttl <= timedelta(0):
            raise ValueError("ttl precisa ser positivo.")
        instant = created_at or datetime.now(UTC)
        created_iso = _iso(instant)
        expires_iso = _iso(instant + effective_ttl)
        scene_digest = hashlib.sha256(scene_data).hexdigest()
        thumbnail_digest = (
            hashlib.sha256(thumbnail_data).hexdigest()
            if thumbnail_data is not None
            else None
        )
        written: list[Path] = []
        removed: list[EvidenceRecord] = []
        record: EvidenceRecord | None = None
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                removed.extend(
                    self._remove_expired_rows(connection, now_iso=created_iso)
                )
                current_size = int(
                    connection.execute(
                        """
                        SELECT COALESCE(SUM(scene_bytes + thumbnail_bytes), 0)
                        FROM evidence_records
                        """
                    ).fetchone()[0]
                )
                required = current_size + total_size - self.max_storage_bytes
                if required > 0:
                    if not self.evict_oldest:
                        raise EvidenceCapacityError(
                            "O armazenamento de evidências atingiu o limite."
                        )
                    removed.extend(self._remove_oldest_rows(connection, required))
                reference = self._new_reference(connection)
                if thumbnail_data is not None:
                    thumbnail_path = self.path_for(reference, variant="thumb")
                    self._write_atomic(thumbnail_path, thumbnail_data)
                    written.append(thumbnail_path)
                scene_path = self.path_for(reference, variant="full")
                self._write_atomic(scene_path, scene_data)
                written.append(scene_path)
                connection.execute(
                    """
                    INSERT INTO evidence_records (
                        reference, scene_sha256, scene_bytes,
                        thumbnail_sha256, thumbnail_bytes, media_type,
                        created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        reference,
                        scene_digest,
                        len(scene_data),
                        thumbnail_digest,
                        len(thumbnail_data) if thumbnail_data is not None else 0,
                        media_type,
                        created_iso,
                        expires_iso,
                    ),
                )
                record = EvidenceRecord(
                    reference=reference,
                    scene_sha256=scene_digest,
                    scene_bytes=len(scene_data),
                    thumbnail_sha256=thumbnail_digest,
                    thumbnail_bytes=(
                        len(thumbnail_data) if thumbnail_data is not None else 0
                    ),
                    media_type=media_type,
                    created_at=instant.astimezone(UTC),
                    expires_at=(instant + effective_ttl).astimezone(UTC),
                )
        except Exception:
            for path in written:
                self._unlink(path)
            raise
        self._delete_files(removed)
        if record is None:  # pragma: no cover
            raise EvidenceStoreError("A evidência não foi persistida.")
        return record

    def get(self, reference: str) -> EvidenceRecord | None:
        safe_reference = self._validate_reference(reference)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_records WHERE reference=?",
                (safe_reference,),
            ).fetchone()
        return self._to_record(row) if row is not None else None

    def list_records(self, *, limit: int = 100, offset: int = 0) -> list[EvidenceRecord]:
        if limit < 1 or limit > 1_000 or offset < 0:
            raise ValueError("Paginação inválida.")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM evidence_records
                ORDER BY created_at DESC, reference
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [self._to_record(row) for row in rows]

    def read(
        self,
        reference: str,
        *,
        variant: Literal["full", "thumb"] = "full",
        now: datetime | None = None,
    ) -> bytes:
        record = self.get(reference)
        if record is None:
            raise EvidenceNotFoundError("Evidência não encontrada.")
        instant = now or datetime.now(UTC)
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("O instante precisa incluir fuso horário.")
        if instant.astimezone(UTC) >= record.expires_at:
            self.delete(record.reference)
            raise EvidenceNotFoundError("Evidência expirada.")
        if variant == "full":
            expected_hash = record.scene_sha256
            expected_size = record.scene_bytes
        elif variant == "thumb":
            expected_hash = record.thumbnail_sha256
            expected_size = record.thumbnail_bytes
            if expected_hash is None:
                raise EvidenceNotFoundError("Miniatura indisponível.")
        else:
            raise ValueError("variant precisa ser full ou thumb.")
        path = self.path_for(record.reference, variant=variant)
        data = self._read_regular_file(path)
        actual_hash = hashlib.sha256(data).hexdigest()
        if len(data) != expected_size or not secrets.compare_digest(
            actual_hash, expected_hash
        ):
            raise EvidenceIntegrityError("A evidência falhou na verificação de integridade.")
        return data

    def verify(self, reference: str) -> bool:
        try:
            record = self.get(reference)
            if record is None:
                return False
            self.read(reference, variant="full", now=record.created_at)
            if record.thumbnail_sha256 is not None:
                self.read(reference, variant="thumb", now=record.created_at)
            return True
        except (EvidenceStoreError, OSError, ValueError):
            return False

    def delete(self, reference: str) -> bool:
        safe_reference = self._validate_reference(reference)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM evidence_records WHERE reference=?",
                (safe_reference,),
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                "DELETE FROM evidence_records WHERE reference=?", (safe_reference,)
            )
            record = self._to_record(row)
        self._delete_files([record])
        return True

    def purge(self, *, now: datetime | None = None) -> PurgeResult:
        instant = now or datetime.now(UTC)
        now_iso = _iso(instant)
        removed: list[EvidenceRecord] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            removed.extend(self._remove_expired_rows(connection, now_iso=now_iso))
            if self.evict_oldest:
                current_size = int(
                    connection.execute(
                        """
                        SELECT COALESCE(SUM(scene_bytes + thumbnail_bytes), 0)
                        FROM evidence_records
                        """
                    ).fetchone()[0]
                )
                if current_size > self.max_storage_bytes:
                    removed.extend(
                        self._remove_oldest_rows(
                            connection, current_size - self.max_storage_bytes
                        )
                    )
        self._delete_files(removed)
        self._reconcile_files(now=instant)
        return PurgeResult(
            records=len(removed),
            bytes_freed=sum(record.storage_bytes for record in removed),
        )

    def total_bytes(self) -> int:
        with self._connect() as connection:
            value = connection.execute(
                """
                SELECT COALESCE(SUM(scene_bytes + thumbnail_bytes), 0)
                FROM evidence_records
                """
            ).fetchone()[0]
        return int(value)

    def path_for(
        self,
        reference: str,
        *,
        variant: Literal["full", "thumb"] = "full",
    ) -> Path:
        safe_reference = self._validate_reference(reference)
        if variant == "full":
            filename = f"{safe_reference}.jpg"
        elif variant == "thumb":
            filename = f"{safe_reference}.thumb.jpg"
        else:
            raise ValueError("variant precisa ser full ou thumb.")
        return self.root / filename

    def _remove_expired_rows(
        self, connection: sqlite3.Connection, *, now_iso: str
    ) -> list[EvidenceRecord]:
        rows = connection.execute(
            """
            SELECT * FROM evidence_records
            WHERE expires_at <= ?
            ORDER BY expires_at, reference
            """,
            (now_iso,),
        ).fetchall()
        if rows:
            connection.executemany(
                "DELETE FROM evidence_records WHERE reference=?",
                [(row["reference"],) for row in rows],
            )
        return [self._to_record(row) for row in rows]

    def _remove_oldest_rows(
        self, connection: sqlite3.Connection, required_bytes: int
    ) -> list[EvidenceRecord]:
        rows = connection.execute(
            """
            SELECT * FROM evidence_records
            ORDER BY created_at, reference
            """
        ).fetchall()
        removed: list[EvidenceRecord] = []
        freed = 0
        for row in rows:
            record = self._to_record(row)
            removed.append(record)
            freed += record.storage_bytes
            if freed >= required_bytes:
                break
        if freed < required_bytes:
            raise EvidenceCapacityError("Não há espaço suficiente para a evidência.")
        connection.executemany(
            "DELETE FROM evidence_records WHERE reference=?",
            [(record.reference,) for record in removed],
        )
        return removed

    def _new_reference(self, connection: sqlite3.Connection) -> str:
        for _ in range(10):
            reference = secrets.token_hex(32)
            exists = connection.execute(
                "SELECT 1 FROM evidence_records WHERE reference=?", (reference,)
            ).fetchone()
            if (
                exists is None
                and not os.path.lexists(self.path_for(reference, variant="full"))
                and not os.path.lexists(self.path_for(reference, variant="thumb"))
            ):
                return reference
        raise EvidenceStoreError("Não foi possível gerar uma referência única.")

    def _write_atomic(self, destination: Path, data: bytes) -> None:
        temporary = self.root / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                os.close(descriptor)
            self._set_private_permissions(temporary, directory=False)
            os.replace(temporary, destination)
            self._set_private_permissions(destination, directory=False)
        except Exception:
            self._unlink(temporary)
            raise

    def _delete_files(self, records: list[EvidenceRecord]) -> None:
        for record in records:
            self._unlink(self.path_for(record.reference, variant="thumb"))
            self._unlink(self.path_for(record.reference, variant="full"))

    def _reconcile_files(self, *, now: datetime) -> None:
        _iso(now)
        if self._is_link_or_reparse(self.root):
            raise EvidenceStoreError("O diretório de evidências não pode ser um link.")
        cutoff = (now.astimezone(UTC) - _TEMPORARY_MAX_AGE).timestamp()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            expected = self._expected_filenames(connection)
            try:
                with os.scandir(self.root) as entries:
                    self._import_legacy_files(
                        connection,
                        entries,
                        expected=expected,
                        now=now.astimezone(UTC),
                    )
                with os.scandir(self.root) as entries:
                    self._remove_unindexed_files(
                        entries,
                        expected=expected,
                        temporary_cutoff=cutoff,
                    )
            except OSError as exc:
                raise EvidenceStoreError(
                    "Não foi possível reconciliar o diretório de evidências."
                ) from exc

    def _import_legacy_files(
        self,
        connection: sqlite3.Connection,
        entries: Iterator[os.DirEntry[str]],
        *,
        expected: set[str],
        now: datetime,
    ) -> None:
        candidates: list[tuple[float, str, str]] = []
        for entry in entries:
            match = _EVIDENCE_FILE_PATTERN.fullmatch(entry.name)
            if match is None or match.group("thumbnail") is not None:
                continue
            reference = match.group("reference").lower()
            canonical_name = f"{reference}.jpg"
            if (
                os.path.normcase(entry.name) != os.path.normcase(canonical_name)
                or os.path.normcase(canonical_name) in expected
            ):
                continue
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if (
                self._metadata_is_link_or_reparse(metadata)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size > self.max_item_bytes
            ):
                continue
            candidates.append((metadata.st_mtime, reference, entry.name))

        current_size = int(
            connection.execute(
                """
                SELECT COALESCE(SUM(scene_bytes + thumbnail_bytes), 0)
                FROM evidence_records
                """
            ).fetchone()[0]
        )
        for modified_at, reference, filename in sorted(candidates, reverse=True):
            try:
                created_at = datetime.fromtimestamp(modified_at, UTC)
                expires_at = created_at + self.ttl
            except (OSError, OverflowError, ValueError):
                continue
            if created_at > now + _MAX_MTIME_SKEW or expires_at <= now:
                continue
            try:
                scene_data = self._read_regular_file(self.root / filename)
                self._validate_jpeg(scene_data, name="cena")
            except (EvidenceStoreError, OSError, ValueError):
                continue
            scene_digest = hashlib.sha256(scene_data).hexdigest()
            if not secrets.compare_digest(scene_digest, reference):
                continue

            thumbnail_data = self._read_legacy_thumbnail(reference)
            if (
                thumbnail_data is not None
                and len(scene_data) + len(thumbnail_data) > self.max_item_bytes
            ):
                thumbnail_data = None
            total_size = len(scene_data) + (
                len(thumbnail_data) if thumbnail_data is not None else 0
            )
            if current_size + total_size > self.max_storage_bytes:
                thumbnail_data = None
                total_size = len(scene_data)
            if (
                len(scene_data) > self.max_item_bytes
                or current_size + total_size > self.max_storage_bytes
            ):
                continue

            thumbnail_digest = (
                hashlib.sha256(thumbnail_data).hexdigest()
                if thumbnail_data is not None
                else None
            )
            connection.execute(
                """
                INSERT INTO evidence_records (
                    reference, scene_sha256, scene_bytes,
                    thumbnail_sha256, thumbnail_bytes, media_type,
                    created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reference,
                    scene_digest,
                    len(scene_data),
                    thumbnail_digest,
                    len(thumbnail_data) if thumbnail_data is not None else 0,
                    "image/jpeg",
                    _iso(created_at),
                    _iso(expires_at),
                ),
            )
            expected.add(os.path.normcase(f"{reference}.jpg"))
            if thumbnail_data is not None:
                expected.add(os.path.normcase(f"{reference}.thumb.jpg"))
                self._set_private_permissions(
                    self.path_for(reference, variant="thumb"),
                    directory=False,
                )
            self._set_private_permissions(
                self.path_for(reference),
                directory=False,
            )
            current_size += total_size

    def _read_legacy_thumbnail(self, reference: str) -> bytes | None:
        path = self.path_for(reference, variant="thumb")
        try:
            metadata = path.lstat()
        except OSError:
            return None
        if (
            self._metadata_is_link_or_reparse(metadata)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > self.max_item_bytes
        ):
            return None
        try:
            data = self._read_regular_file(path)
            self._validate_jpeg(data, name="miniatura")
        except (EvidenceStoreError, OSError, ValueError):
            return None
        return data

    def _remove_unindexed_files(
        self,
        entries: Iterator[os.DirEntry[str]],
        *,
        expected: set[str],
        temporary_cutoff: float,
    ) -> None:
        for entry in entries:
            evidence_match = _EVIDENCE_FILE_PATTERN.fullmatch(entry.name)
            temporary_match = _TEMPORARY_FILE_PATTERN.fullmatch(entry.name)
            if evidence_match is None and temporary_match is None:
                continue
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            unsafe = self._metadata_is_link_or_reparse(metadata)
            if evidence_match is not None:
                if unsafe or os.path.normcase(entry.name) not in expected:
                    self._unlink(self.root / entry.name)
                continue
            if unsafe or (
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_mtime <= temporary_cutoff
            ):
                self._unlink(self.root / entry.name)

    def _expected_filenames(self, connection: sqlite3.Connection) -> set[str]:
        expected: set[str] = set()
        for row in connection.execute("SELECT * FROM evidence_records"):
            try:
                record = self._to_record(row)
                self._validate_index_record(record)
            except (EvidenceStoreError, TypeError, ValueError) as exc:
                raise EvidenceStoreError(
                    "O índice de evidências possui um registro inválido."
                ) from exc
            expected.add(os.path.normcase(f"{record.reference}.jpg"))
            if record.thumbnail_sha256 is not None:
                expected.add(os.path.normcase(f"{record.reference}.thumb.jpg"))
        return expected

    @classmethod
    def _read_regular_file(cls, path: Path) -> bytes:
        try:
            before = path.lstat()
        except OSError as exc:
            raise EvidenceNotFoundError("Arquivo de evidência indisponível.") from exc
        if cls._metadata_is_link_or_reparse(before) or not stat.S_ISREG(
            before.st_mode
        ):
            raise EvidenceIntegrityError(
                "O arquivo de evidência não é um arquivo regular."
            )
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise EvidenceNotFoundError("Arquivo de evidência indisponível.") from exc
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or (
                before.st_dev,
                before.st_ino,
            ) != (
                opened.st_dev,
                opened.st_ino,
            ):
                raise EvidenceIntegrityError(
                    "O arquivo de evidência mudou durante a leitura."
                )
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                return stream.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @staticmethod
    def _unlink(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _to_record(row: sqlite3.Row) -> EvidenceRecord:
        return EvidenceRecord(
            reference=str(row["reference"]),
            scene_sha256=str(row["scene_sha256"]),
            scene_bytes=int(row["scene_bytes"]),
            thumbnail_sha256=(
                str(row["thumbnail_sha256"])
                if row["thumbnail_sha256"] is not None
                else None
            ),
            thumbnail_bytes=int(row["thumbnail_bytes"]),
            media_type=str(row["media_type"]),
            created_at=_parse_time(str(row["created_at"])),
            expires_at=_parse_time(str(row["expires_at"])),
        )

    @classmethod
    def _validate_index_record(cls, record: EvidenceRecord) -> None:
        reference = cls._validate_reference(record.reference)
        if reference != record.reference:
            raise EvidenceStoreError("A referência no índice não é canônica.")
        cls._validate_digest(record.scene_sha256)
        if record.scene_bytes < 1 or record.media_type != "image/jpeg":
            raise EvidenceStoreError("Os metadados da cena são inválidos.")
        if record.thumbnail_sha256 is None:
            if record.thumbnail_bytes != 0:
                raise EvidenceStoreError("Os metadados da miniatura são inválidos.")
        else:
            cls._validate_digest(record.thumbnail_sha256)
            if record.thumbnail_bytes < 1:
                raise EvidenceStoreError("Os metadados da miniatura são inválidos.")
        if record.expires_at <= record.created_at:
            raise EvidenceStoreError("O período de retenção é inválido.")

    @staticmethod
    def _validate_reference(value: str) -> str:
        reference = str(value).strip().lower()
        if len(reference) != 64 or any(character not in _HEX for character in reference):
            raise ValueError("Referência de evidência inválida.")
        return reference

    @staticmethod
    def _validate_digest(value: str) -> None:
        if len(value) != 64 or any(character not in _HEX for character in value):
            raise EvidenceStoreError("Hash inválido no índice de evidências.")

    @staticmethod
    def _validate_jpeg(data: bytes, *, name: str) -> None:
        if len(data) < 6 or not data.startswith(b"\xff\xd8\xff") or not data.endswith(
            b"\xff\xd9"
        ):
            raise ValueError(f"A {name} não é um JPEG válido.")

    @staticmethod
    def _set_private_permissions(path: Path, *, directory: bool) -> None:
        try:
            os.chmod(path, 0o700 if directory else 0o600)
        except OSError:
            pass

    @staticmethod
    def _is_link_or_reparse(path: Path) -> bool:
        try:
            metadata = path.lstat()
        except OSError:
            return True
        return EvidenceStore._metadata_is_link_or_reparse(metadata)

    @staticmethod
    def _metadata_is_link_or_reparse(metadata: os.stat_result) -> bool:
        if stat.S_ISLNK(metadata.st_mode):
            return True
        attributes = getattr(metadata, "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        return bool(attributes & reparse_flag)


__all__ = [
    "EvidenceCapacityError",
    "EvidenceIntegrityError",
    "EvidenceNotFoundError",
    "EvidenceRecord",
    "EvidenceStore",
    "EvidenceStoreError",
    "PurgeResult",
]
