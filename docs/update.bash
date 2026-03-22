#!/bin/bash
# -*- coding: utf-8 -*-

pkg_pyname="aac_datasets"

docs_dpath=`dirname $0`
cd "${docs_dpath}"

rm ${pkg_pyname}*.rst
uv run sphinx-apidoc -e -M -o . "../src/${pkg_pyname}" && uv run make clean && uv run make html

exit 0
