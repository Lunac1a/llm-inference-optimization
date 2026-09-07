#!/usr/bin/env bash
# Reproduce the user-space Python headers used by this WSL baseline; no sudo.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$repo_root/.tmp/python-dev"
mkdir -p "$dest"
cd "$dest"
for package in libpython3.12-dev python3.12-dev; do
    file="${package}_3.12.3-1ubuntu0.16_amd64.deb"
    if [[ ! -f "$file" ]]; then
        apt-get download "$package=3.12.3-1ubuntu0.16"
    fi
done
printf '%s\n' \
  '864360533639b45256258475c7843ac7c56f9bf556b356753f21f7e3012fea67  libpython3.12-dev_3.12.3-1ubuntu0.16_amd64.deb' \
  '841c5ce92fa00027f1cc71af626a92908c372bce90164cae27f5ca961c0ba418  python3.12-dev_3.12.3-1ubuntu0.16_amd64.deb' | sha256sum -c -
mkdir -p extracted
dpkg-deb -x libpython3.12-dev_3.12.3-1ubuntu0.16_amd64.deb extracted
dpkg-deb -x python3.12-dev_3.12.3-1ubuntu0.16_amd64.deb extracted
test -f extracted/usr/include/python3.12/Python.h
