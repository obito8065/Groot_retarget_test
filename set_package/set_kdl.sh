#!/bin/bash
pip install --upgrade pip
pip install catkin_pkg --no-cache-dir -i http://mirrors.jdcloudcs.com/pypi
pip install -e /mnt/workspace/users/zhangtianle/others/pykdl_utils --no-build-isolation
pip install -e /mnt/workspace/users/zhangtianle/others/hrl_geom --no-build-isolation
pip install urdf_parser_py --no-cache-dir -i http://mirrors.jdcloudcs.com/pypi

