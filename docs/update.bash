#!/bin/bash
# -*- coding: utf-8 -*-

pkg_pyname="aac_datasets"

docs_dpath=`dirname $0`
cd "$docs_dpath"

rm ${pkg_pyname}.*rst
sphinx-apidoc -e -M -o . ../src/${pkg_pyname} && make clean && make html

exit 0
