# RunPod setup — EAGS-SLAM on an RTX 3090

Step-by-step setup for running this fork on a RunPod pod. Written from a working Kaggle
setup plus the failure modes actually hit while porting it, so each step notes *why* it is
ordered the way it is.

**Target configuration** (the one the repo was validated against):

| | |
|---|---|
| GPU | RTX 3090, 24 GB, `sm_86` |
| Python | 3.10 (conda env `eags`) |
| PyTorch | 2.1.2 + `cu121` |
| CUDA toolkit | 12.1 (installed **by conda**, not the container) |
| OpenCV | 3.4 + contrib, built from source to `/opt/opencv/34` |

## Do not use PyTorch 2.8 / CUDA 13

DO NOT use CUDA 13:

- `simple-knn`, `diff-gaussian-rasterization-w-pose` and `gaussian_rasterizer` are pinned to
  2023-era commits that predate the `torch.utils.cpp_extension` changes in torch >= 2.2.
  Porting them is a project in itself.
- CUDA 13 removed the older `sm_*` targets these forks hardcode.
- `faiss-gpu=1.8.0` has no CUDA 13 build; you would be forced onto `faiss-cpu`, which is
  slower in the seeding hot path (`mapper_utils.py:208-211`).

**The container's CUDA version is almost irrelevant.** `environment.yml` installs its own
`cuda-toolkit=12.1` and `pytorch-cuda=12.1` from conda, so your runtime CUDA is 12.1 no
matter what the image ships. A CUDA 13 *driver* runs CUDA 12.1 code fine — CUDA is backward
compatible. So pick whichever template RunPod lets you deploy; only the driver matters.

---

## 1. Create the pod

- **GPU:** RTX 3090.
- **Template:** any PyTorch/CUDA template that RunPod will deploy. If a template is rejected
  with *"This GPU will deploy with CUDA 13.x but this template only supports 12.4–12.9"*,
  choose a template that lists CUDA 13.x. Your environment is pinned separately.
- **Container disk: 60 GB minimum.** The default is too small — the OpenCV build alone needs
  ~10 GB of scratch, the conda env ~20 GB, then datasets and outputs.
- **Volume:** `/workspace` is the persistent disk. Put the repo, datasets and outputs there.

**Storage rule: stop the pod, never terminate it.** Stopping preserves the container disk
(so the OpenCV install and conda env survive) while billing only storage. Terminating wipes
it and you repeat steps 2–5. See
[storage types](https://docs.runpod.io/pods/storage/types) and
[network volumes](https://docs.runpod.io/storage/network-volumes).

Layout used below:

```
/workspace/EAGS-SLAM-fork     repo, datasets, outputs   (persistent volume)
/opt/opencv/34                OpenCV install            (container disk)
$CONDA_PREFIX/envs/eags       conda env                 (container disk)
```

## 2. Clone the repo

```bash
cd /workspace
git clone --recursive https://github.com/NaufalGhifari/EAGS-SLAM-fork.git
cd EAGS-SLAM-fork
```

## 3. Create the conda environment

```bash
export CONDA_BASE=$(conda info --base)

conda create -y -n eags -c nvidia/label/cuda-12.1.0 \
    cuda=12.1 cuda-toolkit=12.1 cuda-nvcc=12.1

# MUST be set before the env builds its pip dependencies: the three CUDA extensions are
# compiled during `conda env update`. If the container's CUDA 13 nvcc wins instead, the
# extensions compile against the wrong headers and segfault at runtime.
export CUDA_HOME=$CONDA_BASE/envs/eags
export TORCH_CUDA_ARCH_LIST=8.6          # RTX 3090 = sm_86

conda env update -n eags --file environment.yml --prune
conda activate eags
```

Verify before going further:

```bash
which nvcc && nvcc --version        # must report 12.1, from $CUDA_HOME
cmake --version                     # must be < 3.28 (env pins 3.22)
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

`cmake < 3.28` is important — OpenCV 3.4's build scripts break on newer CMake. That is
why OpenCV is built *after* the env, using the env's CMake 3.22.

`pip install -r requirements.txt` is redundant with `environment.yml` (near-identical
contents, including the same four git packages). Only run it if something failed above.

Then the loop-closure dependency:

```bash
cd thirdparty/Hierarchical-Localization && python -m pip install -e . && cd ../..
```

## 4. Build OpenCV 3.4 + contrib

```bash
mkdir -p /workspace/src && cd /workspace/src
git clone --branch 3.4 --depth 1 https://github.com/opencv/opencv.git
git clone --branch 3.4 --depth 1 https://github.com/opencv/opencv_contrib.git

sudo apt-get update && sudo apt-get install -y build-essential pkg-config libgtk-3-dev

mkdir -p opencv/build && cd opencv/build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/opt/opencv/34 \
      -D OPENCV_EXTRA_MODULES_PATH=/workspace/src/opencv_contrib/modules \
      -D WITH_CUDA=OFF \
      -D WITH_FFMPEG=OFF \
      -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF \
      -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF -D BUILD_DOCS=OFF -D BUILD_opencv_apps=OFF \
      -D ENABLE_PRECOMPILED_HEADERS=OFF \
      ..
make -j$(nproc)
sudo make install

sudo ldconfig
echo "/opt/opencv/34/lib" | sudo tee /etc/ld.so.conf.d/opencv34.conf
sudo ldconfig
ldconfig -p | grep opencv_highgui      # should now list 3.4
```

Flag rationale:

- `CMAKE_INSTALL_PREFIX=/opt/opencv/34` — this exact path is hardcoded in
  `VO/CMakeLists.txt:32` (`find_package(OpenCV 3.4 REQUIRED PATHS /opt/opencv/34)`).
- `WITH_CUDA=OFF` — OpenCV 3.4 cannot build against modern CUDA. Also the single most common
  hard failure.
- `WITH_FFMPEG=OFF` — OpenCV 3.4 predates modern ffmpeg and fails with
  `'CODEC_ID_H264' was not declared`. The VO reads image files, never video, so this is free.
- Python bindings off — only the C++ libraries are needed.

Expected and harmless: `jas_cmshapmat_invmat` stringop-overflow warnings, and
`LIBAVCODEC_VERSION_INT is not defined`. Both are warnings, not errors.

## 5. Build the visual odometry module

`VO/CMakeLists.txt` has a hardcoded conda path that will not exist on RunPod. Edit line 38:

```cmake
# from
set(CONDA_PREFIX "~/.conda/envs/eags")
# to
set(CONDA_PREFIX "$ENV{CONDA_PREFIX}")
```

Then, with the `eags` env active:

```bash
cd VO
mkdir -p build && cd build
cmake -D CMAKE_PREFIX_PATH=$CONDA_PREFIX ..
make -j$(nproc)
cd ../..
```

## 6. Verify everything before running anything long

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# expect: 2.1.2 12.1 True

python -c "import torch; from simple_knn._C import distCUDA2; print('distCUDA2', distCUDA2(torch.rand(1000,3).cuda()).shape)"
# expect: no segfault. If this dies silently, the extensions were built with the wrong nvcc.

python -c "import gaussian_rasterizer, diff_gaussian_rasterization; print('rasterizers OK')"

ldd VO/build/lib/VisualOdom.so | grep "not found" || echo "VO deps OK"
python -c "from VO.build.lib import VisualOdom; print('VO import OK')"
```

`ldd | grep "not found"` is the fastest single check — it lists every unresolved library at
once rather than surfacing them one traceback at a time.

## 7. Headless plotting

```bash
export MPLBACKEND=Agg
evo_config set plot_backend agg
```

Without this, `evo.tools.plot.traj_colormap` fails on a headless host. (The crash is now
non-fatal in this fork — `eval_utils.py` guards the plot — but you still want the PNGs.)

## 8. Run

```bash
%cd /workspace/EAGS-SLAM-fork
python run_slam.py configs/TUM_RGBD/rgbd_dataset_freiburg1_desk.yaml \
    --run_name runpod_baseline 2>&1 | tee log/tum/fr1_desk_runpod.log
```

Notes:

- `config_path` is **positional** — there is no `--config-path` flag.
- Each run writes to `<output_path>/<YYYYMMDD_HHMMSS>_<run_name>/`, so runs never overwrite
  each other. `--overwrite` is required to replace an existing directory.
- Telemetry CSVs land in `<run_dir>/telemetry/`. See `docs/telemetry.md`.

---

## Session restart checklist

After stopping and restarting the same pod, the container disk is intact, so only shell
state and the loader cache need restoring:

```bash
conda activate eags
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST=8.6
export MPLBACKEND=Agg
sudo ldconfig            # re-registers /opt/opencv/34/lib
ldd VO/build/lib/VisualOdom.so | grep "not found" || echo "VO deps OK"
```

If you **terminated** the pod instead, redo steps 2–7 (~40–60 min, mostly the OpenCV build).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `libopencv_highgui.so.3.4: cannot open shared object file` | Loader can't find OpenCV 3.4 | `echo "/opt/opencv/34/lib" \| sudo tee /etc/ld.so.conf.d/opencv34.conf && sudo ldconfig` |
| Same error, but the file exists | Wrong `LD_LIBRARY_PATH` under `conda run` | Prefer `ldconfig` over `export LD_LIBRARY_PATH` — it survives `conda run` |
| `'CODEC_ID_H264' was not declared` | OpenCV 3.4 vs modern ffmpeg | Reconfigure with `-D WITH_FFMPEG=OFF` and check `grep WITH_FFMPEG CMakeCache.txt` says `OFF` |
| OpenCV CMake errors about CMake version | System CMake >= 3.28 | Build OpenCV with the `eags` env activated (CMake 3.22) |
| `distCUDA2` segfaults, no traceback | Extensions built with the container's nvcc | `export CUDA_HOME=$CONDA_PREFIX`, then reinstall the three git packages |
| `main.cpp.o` CMake error, pybind11 not found | `CONDA_PREFIX` hardcoded in `VO/CMakeLists.txt:38` | Edit it to `$ENV{CONDA_PREFIX}` |
| Traceback missing from `conda run` output | `conda run` captures output by default | Add `--no-capture-output` |
| `Unable to determine Axes to steal space for Colorbar` | Headless evo backend | `evo_config set plot_backend agg` (non-fatal in this fork) |
| `unrecognized arguments: --config-path` | `config_path` is positional | `python run_slam.py configs/... --run_name X` |
| ~529 MB download during the first run | Model weights fetched on demand | Needs network access on the first run; cached afterwards |

## What this setup deliberately does not do

- **Does not upgrade PyTorch or CUDA.** See the top of this document.
- **Does not use the container's CUDA toolkit** for anything that matters beyond the driver.
- **Does not install datasets.** Replica/ScanNet are large; fetch them into `/workspace` as
  needed rather than baking them into the image.
