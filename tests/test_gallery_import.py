from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

from app.vision.gallery import (
    JPEG,
    ColumnMapping,
    GalleryImportError,
    GalleryLimitError,
    ImageFormat,
    ImportLimits,
    InvalidGalleryError,
    UnsafeGallerySourceError,
    detect_image_format,
    import_gallery,
)
from scripts.import_incontrol_gallery import main as import_cli


JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00gallery-test\xff\xd9"
PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRgallery-test"
WEBP_BYTES = b"RIFF\x10\x00\x00\x00WEBPVP8 gallery"


def write_zip(path: Path, members: dict[str, bytes], *, compression: int = zipfile.ZIP_DEFLATED) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def load_manifest(output: Path) -> dict[str, object]:
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def test_imports_incontrol_zip_without_copying_credentials(tmp_path: Path) -> None:
    source = tmp_path / "incontrol.zip"
    csv_content = (
        "Código;Nome;Foto;Senha;Cartão;Credencial\r\n"
        "EMP001;José da Silva;fotos\\perfil-001;segredo-123;998877;chave-privada\r\n"
    ).encode("cp1252")
    write_zip(
        source,
        {
            "usuarios.csv": csv_content,
            "fotos/perfil-001": JPEG_BYTES,
        },
    )

    output = tmp_path / "private-gallery"
    result = import_gallery(source, output)

    digest = hashlib.sha256(JPEG_BYTES).hexdigest()
    assert result.imported_count == 1
    assert result.skipped_rows == 0
    assert result.manifest_path == output.resolve() / "manifest.json"
    assert load_manifest(output) == {
        "schema_version": 1,
        "entries": [
            {
                "external_id": "EMP001",
                "display_name": "José da Silva",
                "image_path": f"images/{digest}.jpg",
                "image_sha256": digest,
                "media_type": "image/jpeg",
            }
        ],
    }
    assert (output / "images" / f"{digest}.jpg").read_bytes() == JPEG_BYTES
    assert sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()) == [
        f"images/{digest}.jpg",
        "manifest.json",
    ]
    persisted = b"".join(path.read_bytes() for path in output.rglob("*") if path.is_file())
    assert b"segredo-123" not in persisted
    assert b"998877" not in persisted
    assert b"chave-privada" not in persisted
    assert "segredo-123" not in repr(result)


def test_explicit_mapping_detects_utf16_and_pipe_delimiter(tmp_path: Path) -> None:
    source = tmp_path / "export.zip"
    csv_content = (
        "Chave|Pessoa cadastrada|Retrato|PIN\n"
        "U-77|Márcia Lima|retratos\\u77.png|4321\n"
    ).encode("utf-16")
    write_zip(source, {"dados.txt": csv_content, "retratos/u77.png": PNG_BYTES})

    result = import_gallery(
        source,
        tmp_path / "gallery",
        mapping={
            "id": "Chave",
            "name": "Pessoa cadastrada",
            "photo": "Retrato",
        },
    )

    entry = load_manifest(result.output_dir)["entries"][0]
    assert entry["external_id"] == "U-77"
    assert entry["display_name"] == "Márcia Lima"
    assert entry["media_type"] == "image/png"
    assert "4321" not in (result.manifest_path.read_text(encoding="utf-8"))


def test_detects_utf16_big_endian_without_bom(tmp_path: Path) -> None:
    source = tmp_path / "export.zip"
    table = "ID\tNome\tFoto\n42\tJoão\tface\n".encode("utf-16-be")
    write_zip(source, {"users.csv": table, "face": JPEG_BYTES})

    result = import_gallery(source, tmp_path / "gallery")

    assert load_manifest(result.output_dir)["entries"][0]["display_name"] == "João"


def test_imports_generic_directory_and_photos_without_extensions(tmp_path: Path) -> None:
    source = tmp_path / "photos"
    source.mkdir()
    (source / "EMP-001").write_bytes(WEBP_BYTES)
    (source / "Maria_Souza.jpg").write_bytes(JPEG_BYTES)
    (source / "README.txt").write_text("Galeria local", encoding="utf-8")
    (source / "notes.csv").write_text(
        "Assunto;Descrição\nInstalação;Galeria local\n",
        encoding="utf-8",
    )

    result = import_gallery(source, tmp_path / "gallery")
    entries = load_manifest(result.output_dir)["entries"]

    assert result.imported_count == 2
    assert [(entry["external_id"], entry["display_name"]) for entry in entries] == [
        ("EMP-001", "EMP 001"),
        ("Maria_Souza", "Maria Souza"),
    ]
    assert {entry["media_type"] for entry in entries} == {"image/jpeg", "image/webp"}


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../escape.jpg",
        "/absolute/photo.jpg",
        r"C:\\Windows\\photo.jpg",
        r"\\\\server\\share\\photo.jpg",
        "safe/../../escape.jpg",
        "photo.jpg:alternate-stream",
        "safe//ambiguous.jpg",
    ],
)
def test_rejects_zip_slip_absolute_windows_and_ads_paths(
    tmp_path: Path, unsafe_name: str
) -> None:
    source = tmp_path / "unsafe.zip"
    write_zip(source, {unsafe_name: JPEG_BYTES}, compression=zipfile.ZIP_STORED)

    with pytest.raises(UnsafeGallerySourceError):
        import_gallery(source, tmp_path / "gallery")
    assert not (tmp_path / "gallery").exists()
    assert not (tmp_path / "escape.jpg").exists()


def test_rejects_a_symlink_stored_in_zip(tmp_path: Path) -> None:
    source = tmp_path / "symlink.zip"
    with zipfile.ZipFile(source, "w") as archive:
        link = zipfile.ZipInfo("photos/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "../../outside")

    with pytest.raises(UnsafeGallerySourceError, match="link"):
        import_gallery(source, tmp_path / "gallery")


def test_rejects_case_insensitive_duplicate_zip_members(tmp_path: Path) -> None:
    source = tmp_path / "duplicates.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("photos/face", JPEG_BYTES)
        archive.writestr("PHOTOS/FACE", JPEG_BYTES)

    with pytest.raises(UnsafeGallerySourceError, match="duplicados|ambíguos"):
        import_gallery(source, tmp_path / "gallery")


def test_enforces_file_count_and_compression_ratio_limits(tmp_path: Path) -> None:
    source = tmp_path / "many.zip"
    write_zip(
        source,
        {"one": JPEG_BYTES, "two": PNG_BYTES},
        compression=zipfile.ZIP_STORED,
    )
    with pytest.raises(GalleryLimitError, match="arquivos demais"):
        import_gallery(
            source,
            tmp_path / "count-gallery",
            limits=ImportLimits(max_files=1),
        )

    bomb = tmp_path / "bomb.zip"
    write_zip(bomb, {"large.bin": b"0" * 100_000})
    with pytest.raises(GalleryLimitError, match="compressão"):
        import_gallery(
            bomb,
            tmp_path / "bomb-gallery",
            limits=ImportLimits(max_compression_ratio=2),
        )


def test_rejects_symlinks_in_a_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(JPEG_BYTES)
    try:
        os.symlink(outside, source / "linked.jpg")
    except (OSError, NotImplementedError):
        pytest.skip("O ambiente não permite criar symlink.")

    with pytest.raises(UnsafeGallerySourceError, match="link"):
        import_gallery(source, tmp_path / "gallery")


def test_rejects_sensitive_columns_even_with_explicit_mapping(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    for bad_mapping in (
        ColumnMapping("Senha", "Nome"),
        ColumnMapping("Cartão", "Nome"),
        ColumnMapping("Credencial", "Nome"),
        ColumnMapping("PIN", "Nome"),
        ColumnMapping("Card", "Nome"),
    ):
        with pytest.raises(InvalidGalleryError, match="credenciais"):
            import_gallery(source, tmp_path / bad_mapping.external_id, mapping=bad_mapping)


def test_rejects_unsafe_photo_reference_from_csv(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "face.jpg").write_bytes(JPEG_BYTES)
    (source / "users.csv").write_text(
        "ID;Nome;Foto\n1;Pessoa;C:\\Windows\\secret.jpg\n",
        encoding="utf-8",
    )

    with pytest.raises(UnsafeGallerySourceError, match="absolutos"):
        import_gallery(source, tmp_path / "gallery")


def test_rejects_an_image_extension_with_wrong_magic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "not-a-face.jpg").write_bytes(b"this is not an image")

    with pytest.raises(InvalidGalleryError, match="assinatura"):
        import_gallery(source, tmp_path / "gallery")


def test_reports_an_empty_csv_as_a_gallery_error(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "users.csv").write_bytes(b"")

    with pytest.raises(InvalidGalleryError, match="CSV"):
        import_gallery(source, tmp_path / "gallery")


def test_atomic_import_removes_private_temporary_directory_on_failure(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "person.bin").write_bytes(JPEG_BYTES)
    calls = 0

    def changing_validator(data: bytes) -> ImageFormat | None:
        nonlocal calls
        calls += 1
        return JPEG if calls == 1 else None

    output = tmp_path / "gallery"
    with pytest.raises(InvalidGalleryError, match="mudou"):
        import_gallery(source, output, image_validator=changing_validator)

    assert not output.exists()
    assert not list(tmp_path.glob(".gallery.import-*"))


def test_refuses_to_replace_an_existing_gallery(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "person.jpg").write_bytes(JPEG_BYTES)
    output = tmp_path / "gallery"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(GalleryImportError, match="já existe"):
        import_gallery(source, output)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_cli_outputs_only_sanitized_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "export.zip"
    write_zip(
        source,
        {
            "users.csv": b"ID;Nome;Foto;Senha\n1;Ana;face;do-not-print\n",
            "face": JPEG_BYTES,
        },
    )
    output = tmp_path / "gallery"

    assert import_cli([str(source), str(output)]) == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["imported_count"] == 1
    assert "do-not-print" not in captured.out
    assert "do-not-print" not in captured.err


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (JPEG_BYTES, "image/jpeg"),
        (PNG_BYTES, "image/png"),
        (WEBP_BYTES, "image/webp"),
        (b"not an image", None),
    ],
)
def test_image_detection_uses_magic_bytes(content: bytes, expected: str | None) -> None:
    detected = detect_image_format(content)
    assert (detected.media_type if detected else None) == expected
