# Beatrice v2 backend

The Beatrice backend is optional. The repository does not download, copy, or
redistribute the proprietary runtime, native DLL/PYD files, or model binaries.
RVC remains the default backend and can start without any Beatrice assets.

## Lazy startup

Normal startup never imports the native `beatrice` package and never creates a
converter, Worker, resampler, or FIFO. It performs only a lightweight path and
TOML/bin metadata scan, then shows the GUI with the Gain-only effect chain and
AI disabled. Native validation and model construction happen in the existing
background loader only after **Load Model** is clicked.

`config/local_settings.json` supports an opt-in
`startup.autoload_last_model` flag. It defaults to `false`; when enabled, the
remembered model is scheduled only after the main window has been shown.

## Local configuration

No PowerShell environment variables are required for normal desktop use:

1. Start Voice Changer and select the `BEATRICE` backend.
2. Open **Model Settings** and choose the complete local Runtime folder.
3. Click **Add Model** and choose one complete Beatrice model package folder.
4. Select the package in the Model list, load it, then choose Target Speaker.

Validated paths are saved in the ignored machine-local file
`config/local_settings.json`. Model and runtime files remain in their original
locations; the application records paths and never copies, modifies, deletes,
downloads, or uploads those assets. **Remove From List** only unregisters a
path.

Runtime-folder selection checks only that the expected Python package exists;
it deliberately does not import native code. The full version/API check occurs
with model loading.

Path resolution order is:

- Runtime: GUI/local settings, `VOICE_CHANGE_BEATRICE_RUNTIME_DIR`, then the
  optional project-local `third_party/beatrice_runtime/` folder.
- Models: `models/beatrice/`, GUI-registered package folders, then
  `VOICE_CHANGE_BEATRICE_MODELS_DIR`.

The environment variables remain available for CLI, automated tests, and
advanced users:

```powershell
$env:VOICE_CHANGE_BEATRICE_RUNTIME_DIR = 'D:\path\to\beatrice-runtime'
$env:VOICE_CHANGE_BEATRICE_MODELS_DIR = 'D:\path\to\beatrice-model-packages'
```

The selected runtime root must expose the `beatrice` Python package and
`load_beatrice('2.0.0-rc.0')`. Each immediate child of the model directory is
one selectable package:

```text
<models-dir>/
└── <package>/
    ├── beatrice_paraphernalia_*.toml
    ├── phone_extractor.bin
    ├── pitch_estimator.bin
    ├── embedding_setter.bin
    ├── waveform_generator.bin
    └── speaker_embeddings.bin
```

The loader accepts only runtime/model version `2.0.0-rc.0`, validates the
16 kHz/160-sample input and 24 kHz/240-sample output contract, and checks the
observed native methods and conversion result shape. It does not scan RVC
`.pth` or `.index` files.

## Realtime path

Application audio remains mono float32 at 48 kHz with 256-sample callbacks.
The callback submits blocks to `VoiceConversionWorker`; native inference and
all converter resets happen on its background thread. The adapter keeps a
stateful QQ soxr path (`48 -> 16 -> 24 -> 48 kHz`), native-hop FIFOs, and a
512-sample startup threshold. With the validated runtime this produces 768
samples (about 16 ms) of deterministic startup silence.

Beatrice requires contiguous input. If either Worker queue overflows, the new
block/result is rejected, the current generation is invalidated, queued audio
is discarded, and the Worker thread recreates the converter and resamplers.
The callback emits silence during recovery and falls back to bypass if recovery
cannot complete; it never resets the native runtime itself.

## Assisted tuning

After loading a Beatrice package, open **Model Settings** and choose
**Assisted Tuning**. The dialog reuses the common recording/import, recording
quality, F0 analysis, technical candidate evaluation, playback, and RMS-level
matching components used by RVC, while keeping Beatrice's search policy
separate.

The stages are source-pitch range, pitch coarse, pitch fine, formant, and VQ
neighbors. F0 analysis uses P5/P50/P95. If the selected speaker has TOML
`average_pitch`, its MIDI-note value is converted to Hz and the initial pitch
center is `12 * log2(target/source)`; otherwise the current pitch value is
used. Source-pitch limits are clamped only when the native runtime explicitly
publishes limits. Formant and VQ candidates use the loaded converter's
`max_formant_shift` and `codebook_size`.

Candidate generation pauses the realtime audio stream. Every option owns a
fresh converter, stateful resamplers, and FIFOs, so no candidate can inherit
history from another or from the realtime stream. Cancellation is checked only
after the active native conversion call returns. Candidate WAVs are RMS-matched
for fair playback; automatic checks reject only empty, non-finite, clipped,
silent, discontinuous, or otherwise technically invalid output. The user—not
an automatic similarity score—chooses the preferred voice.

Final values are saved in ignored local settings under a key derived from the
model package metadata identity and target speaker. They are restored when
that speaker is loaded after a restart; model TOML/bin files are never edited.

## Validation

CI-safe tests require no proprietary assets:

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
```

Real integration tests auto-skip when runtime/model paths are unavailable.
The opt-in ten-minute production-path soak is:

```powershell
$env:RUN_BEATRICE_FORMAL_BENCHMARK = '1'
$env:BEATRICE_BENCHMARK_SECONDS = '600'
python -u -m unittest -v tests.test_beatrice_formal_benchmark
```

Before using live audio, verify the target speaker, output level, VB-CABLE
routing, Self Monitor device, and headphones at low volume. Runtime and model
licenses remain the user's responsibility.
