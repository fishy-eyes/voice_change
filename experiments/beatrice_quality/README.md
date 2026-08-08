# Beatrice Audio Quality Root-Cause Analysis

This directory contains offline-only diagnostics. Production modules must not
import it. The tools reuse the production `BeatriceStreamingAdapter` for the
QQ/MQ/HQ paths but never change production quality, startup buffering, model
parameters, or saved application presets.

## Fixed experiment

- Input: `tests/assets/input.wav`
- Package: `jvs`
- Initial target: `jvs001`
- Native path: 16 kHz, continuous 160-sample blocks, direct 24 kHz output
- Streaming path: 48 kHz / 256 samples through the production adapter
- Parameter sweeps: one variable at a time
- Speaker sweep: `jvs001`, `jvs010`, `jvs030`, `jvs050`, `jvs080`

The local model and runtime default to the already ignored research assets in
`experiments/beatrice_probe/assets/`. Override them explicitly when needed.

```powershell
python -m experiments.beatrice_quality.run_quality_matrix
```

After the narrow Source Pitch strategy was retired, generate the four guarded
wide-range assisted-tuning checkpoints with:

```powershell
python -m experiments.beatrice_quality.run_wide_assisted_preview
```

Generated WAV and machine-specific reports are written to ignored `outputs/`
and `results/` directories. The original input is never overwritten.

## Interpretation limits

Waveform correlation, RMSE, spectral energy, and boundary statistics are
diagnostic measurements. They do not prove that one file sounds better. In
particular, Chinese intelligibility and metallic/electronic timbre require the
short human A/B described in `LISTENING_GUIDE.md`.

No proprietary runtime, JVS `.bin`, WAV, JSON result, or local path setting may
be committed from this experiment.
