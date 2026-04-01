import shutil
from pathlib import Path


def copy_with_suffix(src_dir: str, dst_dir: str, suffix: str = "-2"):
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    files = [f for f in src.iterdir() if f.is_file()]
    if not files:
        print("No files found in source directory.")
        return

    for f in files:
        new_name = f.stem + suffix + f.suffix
        dst_file = dst / new_name
        shutil.copy2(f, dst_file)
        print(f"Copied: {f.name} -> {new_name}")

    print(f"\nDone. {len(files)} file(s) copied to '{dst}'.")


if __name__ == "__main__":
    copy_with_suffix("data/Primitives500-2/", "data/Train/", suffix="-2")
