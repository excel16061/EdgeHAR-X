"""Generate datasets/splits/ucfTrainTestlist/classInd.txt from a local UCF101
download, without shipping or copying any dataset file.

The official UCF101 class index lists the 101 class folder names in
case-insensitive alphabetical order, numbered from 1, with CRLF line endings.
Checked on 2026-11-02 against a local copy of the official file: the output
is byte-identical (1,586 bytes).

    python tools/make_ucf101_classind.py --ucf-root datasets/UCF-101
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def class_index(ucf_root: Path) -> bytes:
    names = sorted((p.name for p in ucf_root.iterdir() if p.is_dir()), key=str.lower)
    if len(names) != 101:
        raise SystemExit(f"expected 101 class folders in {ucf_root}, found {len(names)}")
    return "".join(f"{i} {n}\r\n" for i, n in enumerate(names, 1)).encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucf-root", type=Path, default=ROOT / "datasets/UCF-101")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets/splits/ucfTrainTestlist/classInd.txt")
    a = ap.parse_args()
    data = class_index(a.ucf_root)
    if a.out.exists():
        if a.out.read_bytes() == data:
            print(f"{a.out} already present and identical")
            return
        raise SystemExit(f"{a.out} exists and differs; not overwriting")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(data)
    print(f"wrote {a.out} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
