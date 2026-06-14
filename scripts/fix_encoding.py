"""
fix_encoding.py — One-time fix for Windows Unicode encoding errors
===================================================================
Replaces all Unicode symbols in SIA scripts with ASCII equivalents
so they work on Windows cp1252 terminals without crashing.

Run once:
    python fix_encoding.py

Then run normally:
    python train_pipeline.py
"""

import os
import re
from pathlib import Path

# Unicode → ASCII replacements
REPLACEMENTS = [
    ("\u2714", "[OK]"),       # ✔ checkmark
    ("\u2716", "[X]"),        # ✖ cross
    ("\u2718", "[X]"),        # ✘ cross
    ("\u26a0", "[!]"),        # ⚠ warning
    ("\u2192", "->"),         # → arrow
    ("\u2190", "<-"),         # ← arrow
    ("\u2013", "--"),         # – en dash
    ("\u2014", "--"),         # — em dash
    ("\u25ba", ">"),          # ► triangle
    ("\u25c4", "<"),          # ◄ triangle
    ("\u2588", "#"),          # █ block
    ("\u2502", "|"),          # │ pipe
    ("\u251c", "+"),          # ├ tree
    ("\u2500", "-"),          # ─ line
    ("\u2514", "+"),          # └ corner
    ("\u2022", "*"),          # • bullet
    ("\u25c6", "*"),          # ◆ diamond
    ("\u2713", "[OK]"),       # ✓ checkmark
    ("\u2717", "[X]"),        # ✗ cross
    ("\u00e2\u0080\x93", "--"),  # â€" (mangled em dash)
    ("\u00e2\u0080\x94", "--"),  # â€" (mangled em dash variant)
]

# SIA scripts to fix
SIA_SCRIPTS = [
    "sia_eda.py",
    "sia_llm_scoring.py",
    "sia_clustering.py",
    "sia_signals_3_4.py",
    "sia_classifier.py",
    "sia_dossier.py",
    "sia_adversarial.py",
    "train_pipeline.py",
    "predict.py",
    "app.py",
]

UTF8_HEADER = '''\
import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
'''

def already_has_fix(content: str) -> bool:
    return "TextIOWrapper" in content or "PYTHONUTF8" in content

def fix_file(path: Path) -> bool:
    try:
        # Read with utf-8, fallback to latin-1
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1")

        original = content

        # 1. Replace Unicode symbols with ASCII
        for uni, asc in REPLACEMENTS:
            content = content.replace(uni, asc)

        # 2. Fix FileHandler — add encoding="utf-8" if missing
        content = re.sub(
            r'FileHandler\(([^,)]+),\s*mode=["\']w["\']\)',
            r'FileHandler(\1, mode="w", encoding="utf-8")',
            content
        )

        # 3. Add UTF-8 header if not already there (skip train_pipeline — handled separately)
        if not already_has_fix(content) and path.name != "train_pipeline.py":
            # Insert after first docstring or after first import
            lines = content.split("\n")
            insert_at = 0
            in_docstring = False
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Skip shebang / encoding declarations
                if i == 0 and (stripped.startswith("#!") or stripped.startswith("# -*-")):
                    insert_at = 1
                    continue
                # Skip module docstring
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    if in_docstring:
                        insert_at = i + 1
                        in_docstring = False
                        break
                    else:
                        if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                            insert_at = i + 1
                            break
                        in_docstring = True
                        continue
                if in_docstring:
                    continue
                # First non-blank, non-comment line after docstring
                if stripped and not stripped.startswith("#"):
                    insert_at = i
                    break

            lines.insert(insert_at, UTF8_HEADER)
            content = "\n".join(lines)

        if content != original:
            path.write_text(content, encoding="utf-8")
            return True
        return False

    except Exception as e:
        print(f"  [ERROR] Could not fix {path.name}: {e}")
        return False


def main():
    print("=" * 55)
    print("  SIA — Windows Encoding Fix")
    print("=" * 55)

    fixed   = []
    skipped = []
    missing = []

    for name in SIA_SCRIPTS:
        p = Path(name)
        if not p.exists():
            missing.append(name)
            print(f"  [SKIP]  {name} — not found")
            continue

        changed = fix_file(p)
        if changed:
            fixed.append(name)
            print(f"  [FIXED] {name}")
        else:
            skipped.append(name)
            print(f"  [OK]    {name} — already clean")

    print("=" * 55)
    print(f"  Fixed   : {len(fixed)} files")
    print(f"  Already OK : {len(skipped)} files")
    print(f"  Not found  : {len(missing)} files")
    print("=" * 55)
    print()
    print("  Now run:  python train_pipeline.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
