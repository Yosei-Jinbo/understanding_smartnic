project('DOCA_MPI_HOST',
  'c', 'cpp',
  license: 'BSD-3',
  default_options: ['buildtype=release'],
  meson_version: '>= 0.61.2'
)

# 実験APIの警告を抑制（必要なら外してOK）
add_project_arguments('-D', 'DOCA_ALLOW_EXPERIMENTAL_API', language: ['c', 'cpp'])

SAMPLE_NAME = 'comch_server'

# DOCA のライブラリディレクトリ（環境に合わせて変更可）
doca_libdir = '/opt/mellanox/doca/lib/aarch64-linux-gnu'

cc = meson.get_compiler('c')
# libdoca_dma を明示リンク（doca_dma_* 未定義参照の解消）
doca_dma = cc.find_library('doca_dma', dirs: doca_libdir, required: true)
doca_rdma = cc.find_library('doca_rdma', dirs: doca_libdir, required: true)
libm = cc.find_library('m', required: true)

# MPI 依存追加（C から MPI を使う前提）
mpi_dep = dependency('mpi', language: 'c', required: true)

# 依存
sample_dependencies = [
  dependency('doca-common'),
  dependency('doca-comch'),
  dependency('doca-argp'),
  doca_dma,  # ← 追加
  doca_rdma,
  mpi_dep,   # ← これを追加
  libm,
]

# Host 実行バイナリのソース
# - ヘッダは含めない
# - DPU 側の .c は含めない（DPU は別 shared_library）
sample_srcs = [
  SAMPLE_NAME + '.c',
  'rdma_task.c',
  'rs_pool.c',
  'mmap_cache.c',
  'union_tracker.c',
  'collective.c',
  'handlers.c',
  '../common/comch_ctrl_path_common.c',
  '../common/common.c',
  '../common/doca_rdma_utils.c',
  '../common/mpi_exchange.c',
]

# インクルードパス
sample_inc_dirs = [
  include_directories('.'),
  include_directories('..'),
  include_directories('../common'),
]

executable(
  'doca_' + SAMPLE_NAME,
  sample_srcs,
  dependencies: sample_dependencies,
  include_directories: sample_inc_dirs,
  # 実行ファイルにビルド時用の rpath を埋め込む（Meson流儀）
  build_rpath: doca_libdir,
  # インストール時の rpath（インストールしないなら不要だが、将来のために設定）
  install_rpath: doca_libdir,
  install: false
)
