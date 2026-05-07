# setup.py
import os
import shlex
import subprocess
import shutil
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import torch

DEFAULT_DOCA_PKGDIR = "/opt/mellanox/doca/lib/x86_64-linux-gnu/pkgconfig"
DOCA_PKGDIR = os.environ.get("DOCA_PKGCONFIG_DIR", DEFAULT_DOCA_PKGDIR)

DEFAULT_DOCA_LIBDIRS = ["/opt/mellanox/doca/lib/x86_64-linux-gnu"]
EXTRA_DOCA_RPATHS = [p for p in DEFAULT_DOCA_LIBDIRS if os.path.isdir(p)]
TORCH_LIBDIR = os.path.join(os.path.dirname(torch.__file__), "lib")

BAD = ("/usr/lib/x86_64-linux-gnu/openmpi", "/usr/mpi/")

def run_pkg_config(pkgs):
    env = os.environ.copy()
    pieces = []
    if DOCA_PKGDIR:
        pieces.append(DOCA_PKGDIR)
    if env.get("PKG_CONFIG_PATH"):
        pieces.append(env["PKG_CONFIG_PATH"])
    env["PKG_CONFIG_PATH"] = ":".join(pieces)

    def run(args):
        out = subprocess.check_output(args, env=env, text=True).strip()
        return shlex.split(out) if out else []

    cflags = run(["pkg-config", "--cflags"] + list(pkgs))
    libs = run(["pkg-config", "--libs"] + list(pkgs))
    return cflags, libs

def run_mpi_showme():
    mpicxx = shutil.which("mpicxx")
    if not mpicxx:
        raise RuntimeError("mpicxx not found in PATH (set PATH to OMPI6 first)")
    c = subprocess.check_output([mpicxx, "--showme:compile"], text=True).strip()
    l = subprocess.check_output([mpicxx, "--showme:link"], text=True).strip()
    return shlex.split(c) if c else [], shlex.split(l) if l else []

def split_flags(cflags, lflags):
    include_dirs, library_dirs, libraries = [], [], []
    extra_compile_args, extra_link_args = [], []

    def consume(tokens, is_link=False):
        it = iter(tokens)
        for t in it:
            token = t
            if token.startswith("-I"):
                include_dirs.append(token[2:] or next(it))
            elif token.startswith("-L"):
                library_dirs.append(token[2:] or next(it))
            elif token.startswith("-l"):
                libraries.append(token[2:] or next(it))
            else:
                (extra_link_args if is_link else extra_compile_args).append(token)

    consume(cflags, is_link=False)
    consume(lflags, is_link=True)
    return include_dirs, library_dirs, libraries, extra_compile_args, extra_link_args

DOCA_UCX_PKGS = ["doca-common", "doca-comch", "doca-argp", "doca-dma", "doca-rdma"]

cflags_doca, lflags_doca = run_pkg_config(DOCA_UCX_PKGS)
mpi_cflags, mpi_lflags = run_mpi_showme()

cflags_all = cflags_doca + mpi_cflags
lflags_all = lflags_doca + mpi_lflags

inc, libdir, libs, cargs, ldargs = split_flags(cflags_all, lflags_all)

# openmpi4 系の混入を落とす（libdir と ldargs の両方）
libdir = [d for d in libdir if not any(b in d for b in BAD)]
ldargs = [a for a in ldargs if not any(b in a for b in BAD)]

rpaths = [
    f"-Wl,-rpath,{d}"
    for d in set(libdir + EXTRA_DOCA_RPATHS + [TORCH_LIBDIR])
    if os.path.isdir(d)
]

def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() not in ("0", "", "false", "False", "no", "No")

USE_MARCH_NATIVE = env_flag("USE_MARCH_NATIVE", "0")
USE_LTO          = env_flag("USE_LTO", "0")
USE_FAST_MATH    = env_flag("USE_FAST_MATH", "0")

opt_cxx = [
    "-O3", "-DNDEBUG", "-fPIC", "-std=gnu++17",
    "-D", "DOCA_ALLOW_EXPERIMENTAL_API",
    "-DOMPI_SKIP_MPICXX=1",   # ★重要
]
opt_c = [
    "-O3", "-DNDEBUG", "-fPIC",
    "-D", "DOCA_ALLOW_EXPERIMENTAL_API",
]

if USE_MARCH_NATIVE:
    opt_cxx += ["-march=native", "-mtune=native"]
    opt_c   += ["-march=native", "-mtune=native"]

if USE_LTO:
    opt_cxx += ["-flto"]
    opt_c   += ["-flto"]
    ldargs  += ["-flto"]

if USE_FAST_MATH:
    opt_cxx += ["-ffast-math"]
    opt_c   += ["-ffast-math"]

extra_compile_args = {"cxx": opt_cxx + cargs, "c": opt_c + cargs}

ext = CppExtension(
    name="doca_comch_client_pybind",
    sources=[
        "doca_comch_client_pybind.cpp",
        "../common/common.c",
        "../common/rdma_common.c",
        "../common/comch_ctrl_path_common.c",
        "../common/dma_common.c",
        "doca_comch_client_interface.c",
        "comch_client.c",
        "../common/doca_rdma_utils.c",
        "../common/mpi_exchange.c",
    ],
    include_dirs=inc,
    library_dirs=libdir,
    libraries=libs,
    extra_compile_args=extra_compile_args,
    extra_link_args=ldargs + rpaths,
)

setup(
    name="doca_comch_client_pybind",
    version="0.1.0",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)