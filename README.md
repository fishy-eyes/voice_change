# Voice Changer v2.0.0

Windows real-time AI voice changer with two independently managed backends:
RVC and Beatrice v2. The application uses PySide6 for the desktop interface,
`sounddevice` for the 48 kHz mono audio path, and background workers so model
inference does not run inside the PortAudio callback.

This repository is a personal entertainment and learning project. It provides
source code and integration logic; it does not redistribute voice models or the
proprietary Beatrice Runtime.

## Highlights

- RVC and Beatrice v2 backend switching
- Lazy loading: the GUI appears before HuBERT, RMVPE, RVC weights, the Beatrice
  Runtime, or a Beatrice model is loaded
- RVC model import, model profiles, realtime presets, and assisted tuning
- Beatrice Runtime/model folder management and multi-speaker selection
- Beatrice assisted Pitch coarse/fine, Formant, and VQ-neighbor comparisons
- Per-speaker presets with a safe default Source Pitch range of 30–1100 Hz
- Background Worker/StreamingAdapter processing with bounded queues
- Input/output device selection, VB-CABLE routing, and independent SelfMonitor
- English and Chinese desktop interface

## Audio and backend architecture

```text
Microphone / input device
  -> AudioStream (48 kHz mono float32)
  -> VoiceConversionManager
       -> RVC Runtime -> RVCWorker
       -> Beatrice v2 Runtime -> VoiceConversionWorker -> StreamingAdapter
  -> Output Gain
  -> Main output / VB-CABLE
  -> Optional SelfMonitor / headphones
```

`VoiceConversionManager` owns backend selection and serializes model switches.
Only the active runtime owns an AI effect and Worker. Switching or unloading
stops the previous Worker before releasing its model, while the base audio path
and final Output Gain remain available if a model fails to load.

The default startup chain contains only Output Gain. Model discovery reads
lightweight metadata, but native runtimes and model weights are loaded only
after the user chooses a backend/model and clicks **Load Model**. Optional
post-window autoload is disabled by default and can be enabled in the ignored
local settings file.

## Requirements

- Windows 10/11 x64
- Python 3.11
- A working input and output audio device
- NVIDIA CUDA GPU recommended for realtime RVC inference
- VB-CABLE optional for routing converted audio into games or chat software

The validated development stack is described in `environment.yml` and the
pinned requirement files.

## Install and run

Using Conda:

```powershell
conda env create -f environment.yml
conda activate voice_change
python -m pip check
python main.py
```

Or install the runtime dependencies into an existing compatible Python 3.11
environment:

```powershell
python -m pip install -r requirements.txt
python -m pip check
python main.py
```

Development and test dependencies are in `requirements-dev.txt`.

## Configure RVC

The project includes the RVC inference source under `rvc_source/`, but does not
include model weights. Provide these local assets:

For a source checkout, keep the ignored binary assets under:

```text
local_assets/rvc/
├── foundation_models/
│   ├── hubert/
│   │   ├── config.json
│   │   └── pytorch_model.bin
│   └── rmvpe/
│       └── rmvpe.pt
└── voice_models/
    └── <voice>/
        ├── model.pth
        ├── model.index       # optional
        └── profile.json      # optional
```

Import an RVC voice folder from the GUI. It must contain one `.pth` model and
may contain a matching `.index` and `profile.json`. The application registers
the external folder; it does not copy or delete the model files.

Advanced path overrides remain available:

- `VOICE_CHANGE_RVC_SOURCE_DIR`
- `VOICE_CHANGE_RVC_MODELS_DIR`
- `VOICE_CHANGE_RVC_VOICE_MODELS_DIR`
- `VOICE_CHANGE_RVC_DEFAULT_MODEL`

The frozen Windows application intentionally retains its public
`rvc_models/` and `models/rvc/` folders. This keeps the released package and
existing installations compatible while source-development assets stay under
`local_assets/`.

## Configure Beatrice v2

The Beatrice Runtime and Beatrice models are not distributed with this
repository. Users must obtain and supply them separately under their applicable
licenses.

1. Select the **BEATRICE** backend.
2. Open **Model Settings** and choose a Beatrice Runtime folder exposing the
   `beatrice` Python package.
3. Add a complete Beatrice model package folder through the GUI.
4. Select the model and click **Load Model**.
5. Choose the Target Speaker or open **Assisted Tuning**.

The current models use model API architecture `2.0.0-rc.0`. This identifier and
the official `Beatrice20rc0_*` ABI names do not mean that the Runtime
implementation itself must be rc0. See `docs/BEATRICE.md` for the compatibility
matrix, runtime contract, parameters, and local integration tests.

## Local settings and assets

Machine-specific state is stored in `config/local_settings.json`, which is
ignored by Git. The tracked `config/local_settings.example.json` contains only
portable empty defaults.

The repository intentionally excludes:

- RVC `.pth`, `.index`, HuBERT, and RMVPE weights
- Beatrice `.bin` models and proprietary `.pyd`/`.dll` Runtime files
- user Runtime/model paths and customization profiles
- recorded/imported WAV files and generated candidates
- benchmark outputs, result JSON, caches, and logs

Do not commit third-party Runtime or model assets when adding experiments.
Source-checkout RVC defaults and developer Beatrice assets are organized under
the fully ignored `local_assets/` tree described in
`docs/PROJECT_STRUCTURE.md`. The GUI continues to accept arbitrary external
Runtime and voice-model paths.

## Main controls

1. Select input and output devices. Choose VB-CABLE as the main output when the
   destination application should receive converted audio.
2. Select RVC or Beatrice and choose a model.
3. Click **Load Model**, then enable AI Voice.
4. Open **Model Settings** for backend-specific parameters or assisted tuning.
5. Click **Start Audio**.
6. Optionally enable SelfMonitor with a separate headphone output.

SelfMonitor receives the final processed signal but is isolated from the main
output. A monitor failure does not stop VB-CABLE or the selected main output.

## Project layout

```text
ai/                 backend-neutral manager, engines, workers, RVC and Beatrice
audio/              duplex stream, devices, monitoring, and output routing
config/             portable defaults, realtime presets, and local settings API
core/               application runtime/model lifecycle and device switching
customization/      recording analysis, candidate search, evaluation, and profiles
effects/            stable AI adapter, final gain, and ordered effect manager
gui/                PySide6 main window and backend-specific settings panels
docs/                detailed technical documentation
experiments/         source-only probes, methods, and quality tools
local_assets/        ignored developer-local Runtime/model/audio/generated data
models/              packaged-app model placeholders and portable documentation
packaging/           maintained Windows packaging metadata and scripts
rvc_source/          vendored MIT-licensed RVC inference source
tests/               unit, integration, GUI, real-model, and benchmark checks
```

In a source checkout, RVC voice models and shared foundation weights live under
`local_assets/rvc/`. The Windows package keeps `models/rvc/` and
`rvc_models/` as its user-facing installation layout. A Beatrice model may be
registered from `models/beatrice/` or any external folder; its Runtime is a
separate dependency and never belongs under `models/`. See
`docs/PROJECT_STRUCTURE.md` for the complete ownership rules.

## Validation

Run the portable regression suite from the repository root:

```powershell
python -m unittest discover -s tests -p "test_*.py"
python -m compileall -q .
git diff --check
```

Tests requiring real RVC/Beatrice assets, audio hardware, or long benchmarks
are explicit local integration checks and may require environment variables or
project-local ignored assets. A skipped hardware/proprietary test must not be
reported as a successful real-device test.

## Release scope

`v2.0.0` is the first stable multi-backend source baseline. Its GitHub Release
includes the validated Windows x64 application archive, but no proprietary
Beatrice Runtime or user/foundation model files are redistributed.
