# User model directories

This tree is reserved for user-selectable voice-model packages and portable
profile metadata. Real model weights remain local and are ignored by Git.

- `models/rvc/`: RVC voice models (`.pth`, optional `.index`, and profile data).
- `models/beatrice/`: optional Beatrice model packages. The GUI may instead
  register a package from any external folder.

Beatrice Runtime binaries are not models and must not be placed here. RVC's
shared HuBERT and RMVPE foundation weights belong under `rvc_models/`, not
`models/rvc/`.
