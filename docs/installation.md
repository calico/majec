# Installation

MAJEC is a Python package (Python 3.11 or newer) with an optional compiled component. This page covers the install routes, how to choose and confirm the equivalence class (EqC) builder, and the external tools that some helper commands need.

- [Requirements](#requirements)
- [Choosing a builder](#choosing-a-builder)
- [Install with pip](#install-with-pip)
- [Install from source with conda](#install-from-source-with-conda)
- [Install from bioconda](#install-from-bioconda)
- [Building majec-core from source](#building-majec-core-from-source)
- [Confirming which builder runs](#confirming-which-builder-runs)
- [Optional external tools](#optional-external-tools)
- [Verifying the installation](#verifying-the-installation)
- [Upgrading from 0.1.x](#upgrading-from-01x)
- [Troubleshooting](#troubleshooting)

---

## Requirements

- **Python ≥ 3.11.** The conda environment file pins Python 3.11.
- **Linux or macOS.** MAJEC is developed and tested on Linux.
- **Core Python dependencies**, installed automatically: numpy, pandas, scipy, pysam.
- `majec_run_pipeline` reads BAM files directly and needs **no external programs**. featureCounts, samtools and bedtools are not used during quantification.

---

## Choosing a builder

The first step of every run is to read each BAM and group its fragments into EqCs. MAJEC has two builders for this step, and they give the same results:

| Builder | Package | How to get it | Notes |
|---|---|---|---|
| Compiled builder (majec-core) | `majec-core` (Rust) | `pip install "majec[rust]"`, or `pip install majec-core` alongside an existing install | Recommended. Required for the experimental no-NH / long-read options. |
| Pure-Python builder (pysam) | `pysam` | Always installed | Same results; roughly 6× slower on the BAM scan. |

MAJEC uses majec-core automatically when it can be imported, and otherwise falls back to the pure-Python builder. Either way, the per-sample BAM scan is single-threaded; `--threads` controls how many samples are processed in parallel.

The experimental options for aligners without NH tags (`--derive_multiplicity on`, `--lr_score_ratio`, `--lr_min_mapq`) need majec-core. With the pure-Python builder, the pipeline stops with an error rather than ignoring them.

---

## Install with pip

```bash
pip install "majec[rust]"     # compiled builder (majec-core)
```

or, for the pure-Python builder only:

```bash
pip install majec
```

Prebuilt majec-core wheels are built for Linux (x86_64 and aarch64) and macOS on Apple Silicon (arm64), for CPython 3.11, 3.12 and 3.13. Where no wheel matches, pip tries to compile majec-core from source, which needs a Rust toolchain (see [Building majec-core from source](#building-majec-core-from-source)). If that fails, install plain `majec`; results are the same.

### Extras

Extras can be combined, for example `pip install "majec[rust,full_analysis]"`.

| Extra | Installs | Needed for |
|---|---|---|
| `rust` | majec-core | the compiled builder |
| `database` | polars | `majec_build_db` |
| `viz` | plotly, python-kaleido, pypdf2, xlsxwriter | `majec_visualize` (kaleido for SVG/PDF export, pypdf2 for merged PDFs, xlsxwriter for `--export_excel`) |
| `full_analysis` | everything in `database` and `viz` | all downstream tools |

`majec_run_pipeline`, `majec_precompute_annotations`, `majec_add_norm_factors` and `majec_prepare_deseq2` need only the core dependencies. Running the R scripts that `majec_prepare_deseq2` generates needs R with the Bioconductor packages DESeq2 and genefilter.

pip does not install the external programs listed under [Optional external tools](#optional-external-tools). If you need them, install them separately, for example with `mamba install -c conda-forge -c bioconda samtools subread sqlite`.

---

## Install from source with conda

This route gives an isolated environment with pinned versions of every Python dependency and the external tools. We recommend `mamba` (for example from [Miniforge](https://github.com/conda-forge/miniforge)); `conda` works too.

```bash
git clone https://github.com/calico/majec.git
cd majec
mamba env create -f majec.yml
conda activate majec
pip install -e ".[rust]"      # with the compiled builder (prebuilt wheel)
```

For the pure-Python builder only, run `pip install -e .` as the last step instead.

`majec.yml` creates an environment named `majec` containing:

- Python 3.11, numpy, pandas, scipy, polars and pysam;
- the optional visualization libraries: plotly, python-kaleido, pypdf2, xlsxwriter;
- the external tools samtools, Subread (featureCounts) and sqlite.

So no extra beyond `rust` is needed in this environment. The `-e` (editable) install links to the cloned source, so `git pull` updates MAJEC without reinstalling. Rebuild majec-core if its source changes (see below).

Run `conda activate majec` in every new terminal session before using MAJEC.

---

## Install from bioconda

```bash
conda install bioconda::majec
```

The bioconda package is `noarch` and uses the **pure-Python builder (pysam)**. It also installs samtools and Subread for the helper tools. To add the compiled builder, install majec-core from PyPI into the same environment:

```bash
pip install majec-core
```

`majec_build_db` also calls the `sqlite3` command-line program; if it is not on your `PATH`, run `mamba install -c conda-forge sqlite`.

---

## Building majec-core from source

Build majec-core yourself if no prebuilt wheel matches your platform, or if you are working on the Rust code. You need a recent stable Rust toolchain; pip fetches maturin, the build backend, automatically. In a conda environment:

```bash
mamba install -c conda-forge rust maturin
```

(or install Rust with [rustup](https://rustup.rs)). Then, from the repository root:

```bash
pip install ./rust/majec-core
```

This compiles an optimised build and installs the `majec_core` module into the active environment. Re-run the same command after changing anything under `rust/majec-core/`.

---

## Confirming which builder runs

**Before a run**, check whether majec-core can be imported:

```bash
python -c "import majec_core" && pip show majec-core
```

**During a run**, one of the first lines of `{prefix}_pipeline.log` (also printed to the console) names the builder:

```
2026-11-06 10:15:02,114 - INFO -  EqC builder: Rust majec_core 0.2.0 (sha256 a0a66b9495f48f67)
```

The hash is the first 16 hex digits of the SHA-256 of the loaded binary, so it identifies the exact build. With the pure-Python builder, the line is a warning instead:

```
2026-11-06 10:15:02,114 - WARNING -  EqC builder: pure-Python pysam fallback (majec_core not installed). Results are the same; the BAM scan is roughly 6x slower. Install majec-core for the compiled builder.
```

**After a run**, the `eqc_builder` field of `{prefix}_run_manifest.json` records the same information, alongside `majec_version`. The manifest is written when the run uses `--use_cache`.

- With majec-core, `eqc_builder` has `"builder": "rust (majec_core)"` and the `majec_core_version`, `majec_core_binary` (path) and `majec_core_sha256` fields.
- With the pure-Python builder, it has `"builder": "python (pysam fallback)"` and `pysam_version`.

---

## Optional external tools

Only these helper commands use external programs:

| Command | Needs | Purpose |
|---|---|---|
| `majec_calc_frag_len` | [samtools](http://www.htslib.org/) | Fragment-length statistics for the optional `--frag_stats_dir` override. Not needed for paired-end data in 0.2.0, which measures fragment lengths during the BAM scan. |
| `majec_calc_thresholds` | featureCounts from [Subread](http://subread.sourceforge.net/) ≥ 2.0.2 | Cohort-wide expression thresholds for `--use_bins`. |
| `majec_build_db` | the `sqlite3` command-line program | Bulk-loading count matrices into the SQLite database. |

All three are in `majec.yml`. The bioconda package includes samtools and Subread.

---

## Verifying the installation

```bash
majec_run_pipeline --version
majec_run_pipeline --help
```

The first command should print `MAJEC 0.2.0`. The second prints the full option list, starting with:

```
usage: majec_run_pipeline [-h] [--version] [--annotation ANNOTATION]
                          [--bams BAMS [BAMS ...]] [--prefix PREFIX]
...
```

All eight commands are installed: `majec_precompute_annotations`, `majec_run_pipeline`, `majec_build_db`, `majec_add_norm_factors`, `majec_prepare_deseq2`, `majec_visualize`, `majec_calc_frag_len` and `majec_calc_thresholds`.

Next, follow the Quick Start in the [README](../README.md), or see the [Usage Guide](usage_guide.md).

---

## Upgrading from 0.1.x

- Re-run `majec_precompute_annotations` with 0.2.0. Annotation files from 0.1.x still load, but they trigger a slow re-read of the original GTFs, which must still be at their recorded paths.
- Caches written by 0.1.x are not reused. The first 0.2.0 run with `--use_cache` rebuilds them.
- Remove options that no longer exist from saved JSON `--config` files. On the command line they cause an error; in a config file they are silently ignored.
- bedtools and intervaltree are no longer needed and can be removed from your environment.

See [CHANGELOG.md](../CHANGELOG.md) for the full list of changes.

---

## Troubleshooting

**`majec_run_pipeline: command not found`.** The environment where MAJEC is installed is not active. Run `conda activate majec` (or activate your virtualenv) and try again.

**`conda env create` fails with conflicts or download errors.** Use `mamba`, which resolves environments faster and more reliably. Make sure the `conda-forge` and `bioconda` channels are reachable.

**`pip install "majec[rust]"` fails while compiling majec-core.** No prebuilt wheel matched your platform or Python version, and pip fell back to building from source without a Rust toolchain. Install Rust (see [Building majec-core from source](#building-majec-core-from-source)), or install plain `majec` and use the pure-Python builder.

**The log says the pure-Python builder ran, but majec-core is installed.** majec-core is probably installed in a different environment from the one running MAJEC. Run `python -c "import majec_core"` with the same `python` that runs `majec_run_pipeline`.

**`--derive_multiplicity on`, `--lr_score_ratio` or `--lr_min_mapq` stops with "requires the compiled majec_core extension".** These experimental options are implemented only in majec-core. Install it, or drop the options.

**`majec_build_db` fails calling `sqlite3`.** Install the SQLite command-line tool (`mamba install -c conda-forge sqlite`).
