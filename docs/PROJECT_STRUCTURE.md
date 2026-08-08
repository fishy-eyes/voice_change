# Project structure and asset ownership

The source tree separates application code from machine-local dependencies and
generated data. The `v2.0.0` inference architecture and runtime behavior are
unchanged by these ownership rules.

## Tracked source

| Path | Responsibility |
| --- | --- |
| `ai/` | Backend manager, engines, workers, and RVC/Beatrice adapters |
| `audio/` | Realtime audio I/O, devices, monitoring, and routing |
| `config/` | Portable defaults, presets, and local-settings API |
| `core/` | Application lifecycle and runtime/model coordination |
| `customization/` | Recording analysis and assisted parameter search |
| `effects/` | Stable audio-effect adapters and effect ordering |
| `gui/` | PySide6 interface and backend settings panels |
| `rvc_source/` | Vendored MIT-licensed RVC inference implementation |
| `tests/` | Unit, integration, GUI, real-asset, and benchmark checks |
| `experiments/` | Non-production probe/benchmark/quality code and methods |
| `docs/` | Maintained technical documentation |
| `packaging/` | Validated Windows build configuration and release notes |

Production modules never import `experiments/`. Experiments may import
production modules for comparison. Removing the experiment tree must not stop
the desktop application from starting.

## Model and Runtime roles

- `models/rvc/` is the default user RVC voice-model location. Each model has a
  `.pth`, optional `.index`, and optional portable profile metadata.
- `models/beatrice/` is an optional user Beatrice package location. The GUI can
  register any external package folder, so this directory is not mandatory.
- `rvc_models/hubert/` and `rvc_models/rmvpe/` contain the shared foundation
  weights required by RVC inference. They are not user voice models.
- A Beatrice Runtime is executable third-party code, not a model. It stays
  outside `models/` and is selected through the GUI, local settings, or the
  documented environment variable.

The established names are intentionally retained because changing them would
break loaders, profiles, packaging paths, and existing user installations.

## Developer-local assets

The entire `local_assets/` tree is ignored. It is an optional workspace for
assets used during local development and never a required production path:

```text
local_assets/
  beatrice/
    runtimes/
    models/
    audio/
    generated/
  rvc/
    models/
```

The current local Beatrice probe defaults use
`local_assets/beatrice/runtimes/probe-runtime/` and
`local_assets/beatrice/models/jvs/`. Users remain free to keep assets anywhere
and register those paths in the GUI. Real Runtime files, model weights,
recordings, generated WAV files, reports, and machine paths must never be
staged.

## Generated and machine-local state

- `config/local_settings.json` and `config/user_models.json` contain local
  selections and paths and are ignored.
- `build/`, `dist/`, `logs/`, caches, test output, experiment output, and result
  directories are generated and ignored.
- `dist/` may retain local release archives, but no built binary is tracked.
- Proprietary `.pyd`/`.dll`, model `.bin`/`.pth`/`.index`, recorded audio, and
  benchmark JSON are excluded from source control.
