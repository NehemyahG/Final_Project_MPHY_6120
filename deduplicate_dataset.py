"""Detect and remove exact duplicate images from the brain tumor dataset.

The script keeps the first file it sees for each byte-identical image and
removes the rest. It also writes a text report with duplicate counts per folder
so the cleanup is easy to review later.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "Brain_Cancer raw MRI data" / "Brain_Cancer"
REPORT_PATH = PROJECT_ROOT / "outputs_full" / "duplicate_cleanup_report.txt"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def iter_image_files(root: Path) -> List[Path]:
    """Return all image files under the dataset root in a stable order."""
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a content hash for a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_duplicate_plan(files: Sequence[Path]) -> Tuple[Dict[str, List[Path]], List[Path]]:
    """Group identical files by hash and return the files to keep/delete."""
    hash_to_files: DefaultDict[str, List[Path]] = defaultdict(list)
    for path in files:
        hash_to_files[hash_file(path)].append(path)

    duplicates: Dict[str, List[Path]] = {
        digest: paths for digest, paths in hash_to_files.items() if len(paths) > 1
    }

    paths_to_delete: List[Path] = []
    for paths in duplicates.values():
        paths_to_delete.extend(paths[1:])

    return duplicates, paths_to_delete


def format_report(
    total_files: int,
    duplicates: Dict[str, List[Path]],
    deleted_files: Sequence[Path],
    dry_run: bool,
) -> str:
    """Render a concise cleanup report suitable for saving as a text file."""
    deleted_count_by_folder = Counter(path.parent.name for path in deleted_files)
    kept_count = total_files - len(deleted_files)

    lines = [
        f"Dataset root: {DATA_ROOT}",
        f"Dry run: {dry_run}",
        f"Total image files scanned: {total_files}",
        f"Duplicate groups found: {len(duplicates)}",
        f"Files removed: {len(deleted_files)}",
        f"Files kept: {kept_count}",
        "",
        "Duplicates removed per folder:",
    ]

    if deleted_count_by_folder:
        for folder_name in sorted(deleted_count_by_folder):
            lines.append(f"- {folder_name}: {deleted_count_by_folder[folder_name]}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("Duplicate groups:")
    if duplicates:
        for index, (digest, paths) in enumerate(sorted(duplicates.items(), key=lambda item: item[1][0].as_posix()), start=1):
            lines.append(f"{index}. Hash: {digest}")
            lines.append(f"   Kept: {paths[0]}")
            for duplicate_path in paths[1:]:
                lines.append(f"   Removed: {duplicate_path}")
    else:
        lines.append("None found.")

    return "\n".join(lines)


def remove_duplicates(paths_to_delete: Sequence[Path]) -> None:
    """Delete duplicate files from disk."""
    for path in paths_to_delete:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove exact duplicate images from the dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Report duplicates without deleting any files.")
    args = parser.parse_args()

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATA_ROOT}")

    files = iter_image_files(DATA_ROOT)
    duplicates, paths_to_delete = build_duplicate_plan(files)

    if not args.dry_run:
        remove_duplicates(paths_to_delete)

    report = format_report(
        total_files=len(files),
        duplicates=duplicates,
        deleted_files=paths_to_delete,
        dry_run=args.dry_run,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()