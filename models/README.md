# User model directories

This tree documents the public model layout used by the frozen Windows
application. Real model weights remain local and are ignored by Git.

- `models/rvc/`: packaged-app RVC voice models (`.pth`, optional `.index`, and
  profile data).
- `models/beatrice/`: optional Beatrice model packages. The GUI may instead
  register a package from any external folder.

Source-checkout RVC assets belong under `local_assets/rvc/voice_models/` and
`local_assets/rvc/foundation_models/`. Beatrice Runtime binaries are not models
and must not be placed here. The packaged application's shared HuBERT and RMVPE
weights belong under `rvc_models/`, not `models/rvc/`.
