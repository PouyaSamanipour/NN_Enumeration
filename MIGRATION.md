# Migrating Your Existing Repository

This document describes how to reorganise your existing public GitHub repository
so that this package becomes the primary content and the original code is
preserved in a clearly labelled `legacy/` folder.

---

## Overview of the target structure

```
your-repo/
├── relu_region_enumerator/     ← new package (this release)
│   ├── __init__.py
│   ├── bitwise_utils.py
│   ├── core.py
│   └── visualization.py
├── legacy/                     ← your original scripts, preserved as-is
│   ├── README.md               ← brief note explaining these are the old files
│   └── <your original .py files>
├── NN_files/                   ← model files (gitignored by default)
├── run.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── LICENSE
├── CHANGELOG.md
├── .gitignore
└── README.md
```

---

## Step-by-step instructions

### 1. Clone your repo locally (if not already)

```bash
git clone https://github.com/PouyaSamanipour/NN_Enumeration.git
cd <your-repo>
```

### 2. Move your existing files into `legacy/`

```bash
mkdir legacy
git mv *.py legacy/           # moves all top-level Python files
git mv NN_files legacy/       # move model files if they are tracked
# Leave any files you want at the top level (e.g. README, .gitignore)
```

If some files should stay at the root (e.g. an old `run.py`), move only the
ones you want archived:

```bash
git mv Enumeration_module_buffer.py legacy/
git mv Utils_enumeration_bitwise.py legacy/
git mv utils_Enumeration.py legacy/
git mv utils_CSV.py legacy/
git mv plot_res.py legacy/
git mv script.py legacy/
```

### 3. Add a short README inside `legacy/`

Create `legacy/README.md`:

```markdown
# Legacy Code

These are the original scripts prior to packaging.
They are preserved for reference and reproducibility.
The current, installable version of the enumerator lives in the
`relu_region_enumerator/` package at the repository root.
```

### 4. Copy the new package files into the repo root

Copy all files from this zip into the repo root, preserving the folder layout:

```
relu_region_enumerator/   → repo root
run.py                    → repo root
requirements.txt          → repo root
requirements-dev.txt      → repo root
pyproject.toml            → repo root
LICENSE                   → repo root
CHANGELOG.md              → repo root
.gitignore                → repo root  (merge with existing if present)
README.md                 → repo root  (replace or merge)
```

### 5. Stage and commit everything

```bash
git add .
git commit -m "refactor: restructure repo — new installable package, legacy scripts archived"
git push origin main
```

### 6. (Optional) Tag the legacy state first

If you want a clean tag pointing to the last state of the old code before
reorganisation, do this *before* step 2:

```bash
git tag v0.0-legacy
git push origin v0.0-legacy
```

Then anyone can always recover the original code with:

```bash
git checkout v0.0-legacy
```

### 7. (Optional) Update your repo description on GitHub

Go to your repo → Settings (gear icon next to About) and update:
- **Description**: "Exact enumeration of ReLU neural network linear regions via bitwise vertex-adjacency testing."
- **Topics**: `neural-networks`, `relu`, `polytope`, `verification`, `control-systems`, `python`
- **Website**: link to your CDC paper or arXiv preprint once available.

---

## Installing the package from your repo

Once the restructured repo is pushed, anyone can install directly from GitHub:

```bash
pip install git+https://github.com/PouyaSamanipour/NN_Enumeration.git
```

Or clone and install in editable mode for development:

```bash
git clone https://github.com/PouyaSamanipour/NN_Enumeration.git
cd <your-repo>
pip install -e .
```
