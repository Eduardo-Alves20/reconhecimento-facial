from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterator


class GalleryImportError(ValueError):
    """Base exception for failures that leave no partially imported gallery."""


class UnsafeGallerySourceError(GalleryImportError):
    """The input contains an unsafe path, link, special file or archive."""


class GalleryLimitError(GalleryImportError):
    """The input exceeds an explicitly configured resource limit."""


class InvalidGalleryError(GalleryImportError):
    """The input is safe to read, but it is not a usable face gallery."""


@dataclass(frozen=True, slots=True)
class ImportLimits:
    max_files: int = 10_000
    max_total_bytes: int = 1_073_741_824  # 1 GiB, uncompressed.
    max_file_bytes: int = 26_214_400  # 25 MiB.
    max_csv_bytes: int = 16_777_216  # 16 MiB.
    max_rows: int = 100_000
    max_entries: int = 50_000
    max_compression_ratio: float = 250.0

    def __post_init__(self) -> None:
        numeric = (
            self.max_files,
            self.max_total_bytes,
            self.max_file_bytes,
            self.max_csv_bytes,
            self.max_rows,
            self.max_entries,
        )
        if any(value <= 0 for value in numeric) or self.max_compression_ratio <= 0:
            raise ValueError("Todos os limites de importação devem ser positivos.")


@dataclass(frozen=True, slots=True)
class ColumnMapping:
    external_id: str
    display_name: str
    photo: str | None = None


@dataclass(frozen=True, slots=True)
class ImageFormat:
    media_type: str
    extension: str


@dataclass(frozen=True, slots=True)
class GalleryImportResult:
    output_dir: Path
    manifest_path: Path
    imported_count: int
    skipped_rows: int
    warnings: tuple[str, ...] = ()


ImageValidator = Callable[[bytes], ImageFormat | None]

JPEG = ImageFormat("image/jpeg", "jpg")
PNG = ImageFormat("image/png", "png")
WEBP = ImageFormat("image/webp", "webp")
_ALLOWED_FORMATS = {
    (JPEG.media_type, JPEG.extension): JPEG,
    (PNG.media_type, PNG.extension): PNG,
    (WEBP.media_type, WEBP.extension): WEBP,
}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jfif", ".png", ".webp"}
_CSV_SUFFIXES = {".csv", ".txt", ".tsv"}
_STRICT_CSV_SUFFIXES = {".csv", ".tsv"}

_ID_ALIASES = (
    "externalid",
    "userid",
    "idusuario",
    "idpessoa",
    "personid",
    "employeeid",
    "matricula",
    "codigo",
    "codigousuario",
    "codigopessoa",
    "id",
)
_NAME_ALIASES = (
    "displayname",
    "nomecompleto",
    "nomedousuario",
    "nomeusuario",
    "personname",
    "username",
    "nome",
    "name",
)
_PHOTO_ALIASES = (
    "profilephoto",
    "fotoperfil",
    "caminhofoto",
    "arquivoimagem",
    "arquivofoto",
    "photopath",
    "imagepath",
    "imagem",
    "image",
    "foto",
    "photo",
    "avatar",
)
_SENSITIVE_COMPACT_TOKENS = (
    "senha",
    "password",
    "passwd",
    "cartao",
    "cardnumber",
    "cardid",
    "credencial",
    "credential",
    "secret",
    "token",
)


@dataclass(frozen=True, slots=True)
class _InputFile:
    name: str
    size: int
    read_prefix: Callable[[int], bytes]
    read_all: Callable[[], bytes]


@dataclass(frozen=True, slots=True)
class _CsvTable:
    source_name: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    delimiter: str
    encoding: str


@dataclass(frozen=True, slots=True)
class _ResolvedColumns:
    external_id: int
    display_name: int
    photo: int | None


@dataclass(frozen=True, slots=True)
class _PendingEntry:
    external_id: str
    display_name: str
    source_name: str
    image_format: ImageFormat


def detect_image_format(data: bytes) -> ImageFormat | None:
    """Recognise supported image containers by magic bytes, not file extension."""

    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return JPEG
    if len(data) >= 16 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return PNG
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return WEBP
    return None


def import_gallery(
    source: str | Path,
    output_dir: str | Path,
    *,
    mapping: ColumnMapping | Mapping[str, str] | None = None,
    limits: ImportLimits | None = None,
    image_validator: ImageValidator | None = None,
) -> GalleryImportResult:
    """Import an InControl ZIP or a generic directory into a private gallery.

    The source is never extracted wholesale. Only validated images and a minimal
    manifest are written. ``output_dir`` must not already exist; creation is done
    in a private sibling directory and committed with one atomic rename.
    """

    effective_limits = limits or ImportLimits()
    resolved_mapping = _coerce_mapping(mapping)
    source_path = Path(source).expanduser()
    destination = Path(output_dir).expanduser().resolve(strict=False)

    if os.path.lexists(destination):
        raise GalleryImportError("O diretório de saída já existe.")
    if destination.name in {"", ".", ".."}:
        raise GalleryImportError("O diretório de saída é inválido.")

    validator = image_validator or detect_image_format
    with _open_source(source_path, effective_limits) as files:
        files_by_name = {item.name.casefold(): item for item in files}
        image_formats = _classify_images(files, validator, image_validator is not None)
        tables = _load_csv_tables(files, image_formats, effective_limits)
        pending, skipped = _build_pending_entries(
            files,
            files_by_name,
            image_formats,
            tables,
            resolved_mapping,
            effective_limits,
        )
        if not pending:
            raise InvalidGalleryError("Nenhuma pessoa com foto válida foi encontrada.")

        return _commit_gallery(
            destination,
            files_by_name,
            pending,
            skipped,
            validator,
            image_validator is not None,
        )


def _coerce_mapping(
    mapping: ColumnMapping | Mapping[str, str] | None,
) -> ColumnMapping | None:
    if mapping is None:
        return None
    if isinstance(mapping, ColumnMapping):
        result = mapping
    elif isinstance(mapping, Mapping):
        aliases = {
            "external_id": "external_id",
            "id": "external_id",
            "display_name": "display_name",
            "name": "display_name",
            "photo": "photo",
            "image": "photo",
        }
        values: dict[str, str] = {}
        for key, value in mapping.items():
            canonical = aliases.get(str(key))
            if canonical is None or canonical in values or not isinstance(value, str):
                raise InvalidGalleryError("O mapeamento de colunas é inválido.")
            values[canonical] = value
        if "external_id" not in values or "display_name" not in values:
            raise InvalidGalleryError("O mapeamento exige as colunas de ID e nome.")
        result = ColumnMapping(
            external_id=values["external_id"],
            display_name=values["display_name"],
            photo=values.get("photo"),
        )
    else:
        raise InvalidGalleryError("O mapeamento de colunas é inválido.")

    selected = (result.external_id, result.display_name, result.photo)
    if any(value is not None and _is_sensitive_header(value) for value in selected):
        raise InvalidGalleryError("Colunas de credenciais não podem ser importadas.")
    if not result.external_id.strip() or not result.display_name.strip():
        raise InvalidGalleryError("O mapeamento exige as colunas de ID e nome.")
    return result


@contextmanager
def _open_source(source: Path, limits: ImportLimits) -> Iterator[tuple[_InputFile, ...]]:
    if not os.path.lexists(source):
        raise GalleryImportError("A origem da galeria não existe.")
    if _is_link_or_reparse(source):
        raise UnsafeGallerySourceError("Links não são aceitos como origem.")

    if source.is_dir():
        yield _inventory_directory(source, limits)
        return
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise InvalidGalleryError("A origem deve ser um ZIP válido ou um diretório.")

    try:
        with zipfile.ZipFile(source, "r") as archive:
            yield _inventory_zip(archive, limits)
    except GalleryImportError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise UnsafeGallerySourceError("Não foi possível ler o arquivo ZIP com segurança.") from exc


def _inventory_zip(
    archive: zipfile.ZipFile, limits: ImportLimits
) -> tuple[_InputFile, ...]:
    result: list[_InputFile] = []
    seen: set[str] = set()
    total_size = 0
    members = archive.infolist()
    if len(members) > limits.max_files:
        raise GalleryLimitError("A origem possui arquivos demais, incluindo diretórios.")

    for info in members:
        safe_name = _safe_relative_name(info.filename, allow_dot=False)
        collision_key = safe_name.casefold()
        if collision_key in seen:
            raise UnsafeGallerySourceError("O ZIP contém caminhos duplicados ou ambíguos.")
        seen.add(collision_key)

        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(unix_mode):
            raise UnsafeGallerySourceError("O ZIP contém link simbólico.")
        file_type = stat.S_IFMT(unix_mode)
        if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise UnsafeGallerySourceError("O ZIP contém um arquivo especial.")
        if info.flag_bits & 0x1:
            raise UnsafeGallerySourceError("ZIPs criptografados não são aceitos.")
        if info.is_dir() or stat.S_ISDIR(unix_mode):
            continue
        if info.file_size < 0 or info.compress_size < 0:
            raise UnsafeGallerySourceError("O ZIP possui metadados de tamanho inválidos.")
        if info.file_size > limits.max_file_bytes:
            raise GalleryLimitError("Um arquivo excede o limite individual permitido.")
        total_size += info.file_size
        if total_size > limits.max_total_bytes:
            raise GalleryLimitError("O conteúdo excede o limite total permitido.")
        if info.file_size:
            ratio = info.file_size / max(info.compress_size, 1)
            if ratio > limits.max_compression_ratio:
                raise GalleryLimitError("O ZIP excede a taxa máxima de compressão.")

        def read_prefix(length: int, member: zipfile.ZipInfo = info) -> bytes:
            try:
                with archive.open(member, "r") as stream:
                    return stream.read(length)
            except (OSError, RuntimeError, zipfile.BadZipFile, NotImplementedError) as exc:
                raise UnsafeGallerySourceError("Falha ao validar um membro do ZIP.") from exc

        def read_all(member: zipfile.ZipInfo = info) -> bytes:
            try:
                with archive.open(member, "r") as stream:
                    data = stream.read(limits.max_file_bytes + 1)
            except (OSError, RuntimeError, zipfile.BadZipFile, NotImplementedError) as exc:
                raise UnsafeGallerySourceError("Falha ao ler um membro do ZIP.") from exc
            if len(data) > limits.max_file_bytes or len(data) != member.file_size:
                raise GalleryLimitError("Um arquivo diverge do tamanho declarado no ZIP.")
            return data

        result.append(_InputFile(safe_name, info.file_size, read_prefix, read_all))
    return tuple(sorted(result, key=lambda item: item.name.casefold()))


def _inventory_directory(source: Path, limits: ImportLimits) -> tuple[_InputFile, ...]:
    root = source.resolve(strict=True)
    result: list[_InputFile] = []
    seen: set[str] = set()
    total_size = 0
    entry_count = 0

    def raise_walk_error(error: OSError) -> None:
        raise UnsafeGallerySourceError("Não foi possível percorrer a origem com segurança.") from error

    for current_root, dir_names, file_names in os.walk(
        root,
        topdown=True,
        onerror=raise_walk_error,
        followlinks=False,
    ):
        current = Path(current_root)
        for directory_name in list(dir_names):
            entry_count += 1
            if entry_count > limits.max_files:
                raise GalleryLimitError("A origem possui arquivos demais, incluindo diretórios.")
            directory = current / directory_name
            if _is_link_or_reparse(directory):
                raise UnsafeGallerySourceError("A origem contém link ou ponto de junção.")

        for file_name in file_names:
            entry_count += 1
            if entry_count > limits.max_files:
                raise GalleryLimitError("A origem possui arquivos demais, incluindo diretórios.")
            path = current / file_name
            if _is_link_or_reparse(path):
                raise UnsafeGallerySourceError("A origem contém link simbólico.")
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise UnsafeGallerySourceError("Não foi possível validar um arquivo da origem.") from exc
            if not stat.S_ISREG(metadata.st_mode):
                raise UnsafeGallerySourceError("A origem contém um arquivo especial.")
            try:
                resolved = path.resolve(strict=True)
                relative = resolved.relative_to(root).as_posix()
            except (OSError, ValueError) as exc:
                raise UnsafeGallerySourceError("Um arquivo aponta para fora da origem.") from exc

            safe_name = _safe_relative_name(relative, allow_dot=False)
            collision_key = safe_name.casefold()
            if collision_key in seen:
                raise UnsafeGallerySourceError("A origem contém caminhos ambíguos.")
            seen.add(collision_key)
            if metadata.st_size > limits.max_file_bytes:
                raise GalleryLimitError("Um arquivo excede o limite individual permitido.")
            total_size += metadata.st_size
            if total_size > limits.max_total_bytes:
                raise GalleryLimitError("O conteúdo excede o limite total permitido.")

            identity = (metadata.st_dev, metadata.st_ino, metadata.st_size)

            def read_prefix(
                length: int, item: Path = path, expected: tuple[int, int, int] = identity
            ) -> bytes:
                return _read_local_file(item, expected, length=length, full=False)

            def read_all(
                item: Path = path, expected: tuple[int, int, int] = identity
            ) -> bytes:
                return _read_local_file(
                    item,
                    expected,
                    length=limits.max_file_bytes + 1,
                    full=True,
                )

            result.append(_InputFile(safe_name, metadata.st_size, read_prefix, read_all))
    return tuple(sorted(result, key=lambda item: item.name.casefold()))


def _read_local_file(
    path: Path,
    expected: tuple[int, int, int],
    *,
    length: int,
    full: bool,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise UnsafeGallerySourceError("Um arquivo da origem mudou durante a importação.") from exc
    try:
        metadata = os.fstat(descriptor)
        current = (metadata.st_dev, metadata.st_ino, metadata.st_size)
        if not stat.S_ISREG(metadata.st_mode) or current != expected:
            raise UnsafeGallerySourceError("Um arquivo da origem mudou durante a importação.")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(length)
        if full and (len(data) != expected[2] or len(data) >= length):
            raise GalleryLimitError("Um arquivo excede ou diverge do tamanho validado.")
        return data
    finally:
        os.close(descriptor)


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise UnsafeGallerySourceError("Não foi possível validar o caminho informado.") from exc
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _safe_relative_name(value: str, *, allow_dot: bool) -> str:
    if not value or "\x00" in value:
        raise UnsafeGallerySourceError("A origem contém um caminho inválido.")
    windows = PureWindowsPath(value)
    replaced = value.replace("\\", "/")
    if windows.drive or windows.root or replaced.startswith(("/", "//")):
        raise UnsafeGallerySourceError("Caminhos absolutos não são aceitos.")
    raw_parts = replaced.split("/")
    if any(part == ".." for part in raw_parts):
        raise UnsafeGallerySourceError("A origem tenta acessar um diretório externo.")
    if not allow_dot:
        unambiguous_parts = raw_parts[:-1] if raw_parts[-1] == "" else raw_parts
        if any(part in {"", "."} for part in unambiguous_parts):
            raise UnsafeGallerySourceError("A origem contém um caminho ambíguo.")
    parts = [part for part in raw_parts if part not in {"", "."}]
    if not parts or any(":" in part for part in parts):
        raise UnsafeGallerySourceError("A origem contém um caminho inválido.")
    normalised = PurePosixPath(*parts).as_posix()
    if normalised in {".", ".."}:
        raise UnsafeGallerySourceError("A origem contém um caminho inválido.")
    return normalised


def _classify_images(
    files: Sequence[_InputFile],
    validator: ImageValidator,
    custom_validator: bool,
) -> dict[str, ImageFormat]:
    result: dict[str, ImageFormat] = {}
    invalid_named_image = False
    for item in files:
        suffix = PurePosixPath(item.name).suffix.casefold()
        sample = item.read_all() if custom_validator else item.read_prefix(32)
        try:
            detected = validator(sample)
        except Exception as exc:
            raise InvalidGalleryError("O validador de imagem recusou a origem.") from exc
        if detected is not None:
            detected = _validate_image_format(detected)
            result[item.name.casefold()] = detected
        elif suffix in _IMAGE_SUFFIXES:
            invalid_named_image = True
    if invalid_named_image:
        raise InvalidGalleryError("A origem contém arquivo de imagem com assinatura inválida.")
    return result


def _validate_image_format(image_format: ImageFormat) -> ImageFormat:
    if not isinstance(image_format, ImageFormat):
        raise InvalidGalleryError("O validador retornou um formato de imagem inválido.")
    canonical = _ALLOWED_FORMATS.get((image_format.media_type, image_format.extension))
    if canonical is None:
        raise InvalidGalleryError("O formato de imagem retornado não é permitido.")
    return canonical


def _load_csv_tables(
    files: Sequence[_InputFile],
    image_formats: Mapping[str, ImageFormat],
    limits: ImportLimits,
) -> tuple[_CsvTable, ...]:
    tables: list[_CsvTable] = []
    for item in files:
        if item.name.casefold() in image_formats:
            continue
        suffix = PurePosixPath(item.name).suffix.casefold()
        csv_candidate = suffix in _CSV_SUFFIXES
        strict_csv = suffix in _STRICT_CSV_SUFFIXES
        if item.size > limits.max_csv_bytes:
            if strict_csv:
                raise GalleryLimitError("Um CSV excede o limite permitido.")
            continue
        if not csv_candidate:
            prefix = item.read_prefix(4_096)
            if b"\n" not in prefix and b"\r" not in prefix:
                continue
            if not any(delimiter in prefix for delimiter in (b",", b";", b"\t", b"|")):
                continue
        try:
            table = _parse_csv(item.name, item.read_all(), limits)
        except InvalidGalleryError:
            if strict_csv:
                raise
            continue
        tables.append(table)
    return tuple(tables)


def _parse_csv(name: str, raw: bytes, limits: ImportLimits) -> _CsvTable:
    text, encoding = _decode_csv(raw)
    sample = text[:65_536]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        counts = {candidate: first_line.count(candidate) for candidate in ",;\t|"}
        delimiter = max(counts, key=counts.get)
        if counts[delimiter] == 0:
            raise InvalidGalleryError("O CSV não possui um delimitador reconhecido.")

    try:
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
        raw_headers = next(reader)
        headers = tuple(_clean_header(value) for value in raw_headers)
        if len(headers) < 2 or any(not value for value in headers):
            raise InvalidGalleryError("O cabeçalho do CSV é inválido.")
        normalised = [_normalise_header(value) for value in headers]
        if len(set(normalised)) != len(normalised):
            raise InvalidGalleryError("O CSV possui colunas duplicadas ou ambíguas.")

        rows: list[tuple[str, ...]] = []
        for raw_row in reader:
            if not raw_row or not any(value.strip() for value in raw_row):
                continue
            if len(rows) >= limits.max_rows:
                raise GalleryLimitError("O CSV possui registros demais.")
            padded = (raw_row + [""] * len(headers))[: len(headers)]
            rows.append(tuple(padded))
    except (csv.Error, UnicodeError, StopIteration) as exc:
        raise InvalidGalleryError("O CSV está malformado.") from exc
    return _CsvTable(name, headers, tuple(rows), delimiter, encoding)


def _decode_csv(raw: bytes) -> tuple[str, str]:
    attempts: list[tuple[str, str]] = []
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        attempts.append(("utf-16", "utf-16"))
    elif raw.startswith(b"\xef\xbb\xbf"):
        attempts.append(("utf-8-sig", "utf-8-sig"))
    else:
        if raw[:4].count(0) >= 1:
            sample = raw[:256]
            even_nulls = sample[0::2].count(0)
            odd_nulls = sample[1::2].count(0)
            utf16_attempts = (
                (("utf-16-be", "utf-16-be"), ("utf-16-le", "utf-16-le"))
                if even_nulls > odd_nulls
                else (("utf-16-le", "utf-16-le"), ("utf-16-be", "utf-16-be"))
            )
            attempts.extend(utf16_attempts)
        attempts.extend((("utf-8-sig", "utf-8"), ("cp1252", "cp1252"), ("latin-1", "latin-1")))
    for codec, label in attempts:
        try:
            value = raw.decode(codec, errors="strict")
        except (UnicodeDecodeError, LookupError):
            continue
        if "\x00" in value:
            continue
        return value.lstrip("\ufeff"), label
    raise InvalidGalleryError("Não foi possível detectar a codificação do CSV.")


def _build_pending_entries(
    files: Sequence[_InputFile],
    files_by_name: Mapping[str, _InputFile],
    image_formats: Mapping[str, ImageFormat],
    tables: Sequence[_CsvTable],
    mapping: ColumnMapping | None,
    limits: ImportLimits,
) -> tuple[tuple[_PendingEntry, ...], int]:
    has_compatible_table = any(
        _resolve_columns(table.headers, mapping) is not None for table in tables
    )
    if tables and has_compatible_table:
        table, columns = _select_table(tables, mapping)
        return _entries_from_table(
            table,
            columns,
            files_by_name,
            image_formats,
            limits,
        )
    if mapping is not None:
        raise InvalidGalleryError("Nenhum CSV compatível com o mapeamento foi encontrado.")
    return _entries_from_images(files, image_formats, limits), 0


def _select_table(
    tables: Sequence[_CsvTable], mapping: ColumnMapping | None
) -> tuple[_CsvTable, _ResolvedColumns]:
    candidates: list[tuple[int, _CsvTable, _ResolvedColumns]] = []
    for table in tables:
        columns = _resolve_columns(table.headers, mapping)
        if columns is None:
            continue
        compact_name = _normalise_header(PurePosixPath(table.source_name).stem)
        score = 6 + (4 if columns.photo is not None else 0)
        if any(token in compact_name for token in ("usuario", "user", "pessoa", "person")):
            score += 3
        if any(token in compact_name for token in ("senha", "password", "credencial", "card", "cartao")):
            score -= 10
        candidates.append((score, table, columns))

    if not candidates:
        raise InvalidGalleryError("Nenhum CSV de pessoas com colunas de ID e nome foi encontrado.")
    candidates.sort(key=lambda item: (-item[0], item[1].source_name.casefold()))
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise InvalidGalleryError("Mais de um CSV de pessoas é igualmente compatível.")
    _, table, columns = candidates[0]
    return table, columns


def _resolve_columns(
    headers: Sequence[str], mapping: ColumnMapping | None
) -> _ResolvedColumns | None:
    normalised = [_normalise_header(value) for value in headers]
    if mapping is not None:
        external_id = _find_header(headers, normalised, mapping.external_id)
        display_name = _find_header(headers, normalised, mapping.display_name)
        photo = _find_header(headers, normalised, mapping.photo) if mapping.photo else None
        if external_id is None or display_name is None or (mapping.photo and photo is None):
            return None
    else:
        external_id = _find_alias(normalised, _ID_ALIASES)
        display_name = _find_alias(normalised, _NAME_ALIASES)
        photo = _find_alias(normalised, tuple(value.casefold() for value in _PHOTO_ALIASES))
        if external_id is None or display_name is None:
            return None

    selected = [headers[external_id], headers[display_name]]
    if photo is not None:
        selected.append(headers[photo])
    if any(_is_sensitive_header(value) for value in selected):
        raise InvalidGalleryError("Colunas de credenciais não podem ser importadas.")
    return _ResolvedColumns(external_id, display_name, photo)


def _find_header(
    headers: Sequence[str], normalised: Sequence[str], requested: str | None
) -> int | None:
    if requested is None:
        return None
    exact = [index for index, value in enumerate(headers) if value.casefold() == requested.strip().casefold()]
    if len(exact) == 1:
        return exact[0]
    target = _normalise_header(requested)
    matches = [index for index, value in enumerate(normalised) if value == target]
    return matches[0] if len(matches) == 1 else None


def _find_alias(headers: Sequence[str], aliases: Sequence[str]) -> int | None:
    for alias in aliases:
        matches = [index for index, value in enumerate(headers) if value == alias.casefold()]
        if len(matches) == 1:
            return matches[0]
    return None


def _entries_from_table(
    table: _CsvTable,
    columns: _ResolvedColumns,
    files_by_name: Mapping[str, _InputFile],
    image_formats: Mapping[str, ImageFormat],
    limits: ImportLimits,
) -> tuple[tuple[_PendingEntry, ...], int]:
    image_names = tuple(
        files_by_name[key].name for key in sorted(image_formats) if key in files_by_name
    )
    by_basename: dict[str, list[str]] = {}
    by_stem: dict[str, list[str]] = {}
    for name in image_names:
        path = PurePosixPath(name)
        by_basename.setdefault(path.name.casefold(), []).append(name)
        stem_key = _normalise_identity(path.stem if path.suffix else path.name)
        by_stem.setdefault(stem_key, []).append(name)

    pending: list[_PendingEntry] = []
    skipped = 0
    identity_names: dict[str, str] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for row in table.rows:
        external_id = _clean_value(row[columns.external_id], maximum=128)
        display_name = _clean_value(row[columns.display_name], maximum=256)
        if not external_id or not display_name:
            skipped += 1
            continue

        source_name: str | None = None
        if columns.photo is not None and row[columns.photo].strip():
            reference = _safe_relative_name(row[columns.photo].strip(), allow_dot=True)
            source_name = _resolve_photo_reference(
                reference, files_by_name, by_basename, by_stem
            )
        if source_name is None:
            source_name = _infer_photo(external_id, display_name, by_stem)
        if source_name is None or source_name.casefold() not in image_formats:
            skipped += 1
            continue

        identity_key = external_id.casefold()
        prior_name = identity_names.setdefault(identity_key, display_name)
        if prior_name != display_name:
            raise InvalidGalleryError("Um mesmo ID está associado a nomes diferentes.")
        pair = (identity_key, source_name.casefold())
        if pair in seen_pairs:
            skipped += 1
            continue
        seen_pairs.add(pair)
        pending.append(
            _PendingEntry(external_id, display_name, source_name, image_formats[source_name.casefold()])
        )
        if len(pending) > limits.max_entries:
            raise GalleryLimitError("A galeria possui pessoas/fotos demais.")
    return tuple(pending), skipped


def _resolve_photo_reference(
    reference: str,
    files_by_name: Mapping[str, _InputFile],
    by_basename: Mapping[str, list[str]],
    by_stem: Mapping[str, list[str]],
) -> str | None:
    direct = files_by_name.get(reference.casefold())
    if direct is not None:
        return direct.name
    basename = PurePosixPath(reference).name.casefold()
    basename_matches = by_basename.get(basename, [])
    if len(basename_matches) == 1:
        return basename_matches[0]
    path = PurePosixPath(reference)
    stem_matches = by_stem.get(_normalise_identity(path.stem if path.suffix else path.name), [])
    return stem_matches[0] if len(stem_matches) == 1 else None


def _infer_photo(
    external_id: str,
    display_name: str,
    by_stem: Mapping[str, list[str]],
) -> str | None:
    for value in (external_id, display_name):
        matches = by_stem.get(_normalise_identity(value), [])
        if len(matches) == 1:
            return matches[0]
    return None


def _entries_from_images(
    files: Sequence[_InputFile],
    image_formats: Mapping[str, ImageFormat],
    limits: ImportLimits,
) -> tuple[_PendingEntry, ...]:
    pending: list[_PendingEntry] = []
    identifiers: set[str] = set()
    for item in files:
        image_format = image_formats.get(item.name.casefold())
        if image_format is None:
            continue
        path = PurePosixPath(item.name)
        identifier = _clean_value(path.stem if path.suffix else path.name, maximum=128)
        if not identifier or identifier.casefold() in identifiers:
            raise InvalidGalleryError("As fotos sem CSV não possuem IDs únicos.")
        identifiers.add(identifier.casefold())
        display_name = re.sub(r"[_-]+", " ", identifier).strip() or identifier
        pending.append(_PendingEntry(identifier, display_name, item.name, image_format))
        if len(pending) > limits.max_entries:
            raise GalleryLimitError("A galeria possui pessoas/fotos demais.")
    return tuple(pending)


def _commit_gallery(
    destination: Path,
    files_by_name: Mapping[str, _InputFile],
    pending: Sequence[_PendingEntry],
    skipped: int,
    validator: ImageValidator,
    custom_validator: bool,
) -> GalleryImportResult:
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.import-", dir=parent))
    _set_private_permissions(temporary, directory=True)
    committed = False
    try:
        images_dir = temporary / "images"
        images_dir.mkdir(mode=0o700)
        entries: list[dict[str, str]] = []
        written_hashes: dict[str, tuple[str, ImageFormat]] = {}

        for item in pending:
            source_file = files_by_name[item.source_name.casefold()]
            data = source_file.read_all()
            try:
                actual = validator(data if custom_validator else data[:32])
            except Exception as exc:
                raise InvalidGalleryError("Uma imagem mudou ou falhou na validação final.") from exc
            if actual is None or _validate_image_format(actual) != item.image_format:
                raise InvalidGalleryError("Uma imagem mudou ou possui formato inconsistente.")

            digest = hashlib.sha256(data).hexdigest()
            existing = written_hashes.get(digest)
            if existing is None:
                relative_path = f"images/{digest}.{item.image_format.extension}"
                image_path = temporary / relative_path
                _write_private_file(image_path, data)
                written_hashes[digest] = (relative_path, item.image_format)
            else:
                relative_path, existing_format = existing
                if existing_format != item.image_format:
                    raise InvalidGalleryError("Uma imagem possui formatos conflitantes.")

            entries.append(
                {
                    "external_id": item.external_id,
                    "display_name": item.display_name,
                    "image_path": relative_path,
                    "image_sha256": digest,
                    "media_type": item.image_format.media_type,
                }
            )

        entries.sort(
            key=lambda value: (
                value["external_id"].casefold(),
                value["display_name"].casefold(),
                value["image_sha256"],
            )
        )
        manifest = {"schema_version": 1, "entries": entries}
        manifest_bytes = (
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        _write_private_file(temporary / "manifest.json", manifest_bytes)
        _set_private_permissions(images_dir, directory=True)
        _set_private_permissions(temporary, directory=True)

        if os.path.lexists(destination):
            raise GalleryImportError("O diretório de saída passou a existir durante a importação.")
        os.replace(temporary, destination)
        committed = True
        _set_private_permissions(destination, directory=True)
    except Exception:
        if not committed:
            shutil.rmtree(temporary, ignore_errors=True)
        raise

    warnings: tuple[str, ...] = ()
    if skipped:
        warnings = (f"{skipped} registro(s) sem foto utilizável foram ignorados.",)
    return GalleryImportResult(
        output_dir=destination,
        manifest_path=destination / "manifest.json",
        imported_count=len(entries),
        skipped_rows=skipped,
        warnings=warnings,
    )


def _write_private_file(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    _set_private_permissions(path, directory=False)


def _set_private_permissions(path: Path, *, directory: bool) -> None:
    try:
        path.chmod(0o700 if directory else 0o600)
    except OSError:
        # Windows ACLs are inherited from the private parent; chmod is best effort.
        pass


def _clean_header(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip().lstrip("\ufeff")


def _normalise_header(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(character.casefold() for character in decomposed if character.isalnum())


def _normalise_identity(value: str) -> str:
    return _normalise_header(value)


def _is_sensitive_header(value: str) -> bool:
    compact = _normalise_header(value)
    if compact in {"pin", "card", "rfid", "badge"}:
        return True
    return any(token in compact for token in _SENSITIVE_COMPACT_TOKENS)


def _clean_value(value: str, *, maximum: int) -> str:
    normalised = unicodedata.normalize("NFC", value).strip()
    if not normalised or len(normalised) > maximum:
        return ""
    if any(unicodedata.category(character).startswith("C") for character in normalised):
        return ""
    return normalised


__all__ = [
    "ColumnMapping",
    "GalleryImportError",
    "GalleryImportResult",
    "GalleryLimitError",
    "ImageFormat",
    "ImportLimits",
    "InvalidGalleryError",
    "JPEG",
    "PNG",
    "UnsafeGallerySourceError",
    "WEBP",
    "detect_image_format",
    "import_gallery",
]
