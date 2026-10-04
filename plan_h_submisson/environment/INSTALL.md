# Environment

Model/data code was developed and checked on Python3.14.4, Linux x86_64, with numerical-library CPU threads capped at2. Exact installed versions are in requirements.lock.txt. `bash scripts/setup_env.sh` creates a private environment; it never changes system Python. If Python3.14 is unavailable, install it in a user-managed location and set `PYTHON=/path/python3.14` for setup. Compatibility on other versions is not asserted.

The bundled Tectonic Linux x86_64 binary and license come from the original manuscript package. `scripts/build_papers.sh` builds source with it (or TECTONIC override). Tectonic obtains its TeX bundle over the network on first use and uses an ordinary local cache. Required TeX components include ICML2026, xeCJK, fontspec, AMS packages, booktabs, graphicx, algorithm, tabularx, hyperref and natbib. The NotoSansCJKkr font and its license are in paper/fonts; Korean uses the prior xeCJK configuration. Source compiler logs and bundle metadata are retained; no full cache/venv is included.

All detector/model-source assets are local to the extracted archive. Historical source has documentary original paths but is not part of default command execution. No GPU or paid service is required.
