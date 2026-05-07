#!/bin/bash
set -e

rm -rf /tmp/build
meson setup /tmp/build -Dbuildtype=release
ninja -C /tmp/build