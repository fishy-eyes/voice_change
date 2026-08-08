Voice Changer v2.0.0 - Windows x64
====================================

Start
-----
Run VoiceChanger.exe. The application can start with the base effects before
backend assets are installed. Runtime logs are written to logs\voice_change.log.

Required RVC assets
-------------------
The public release intentionally does not redistribute model weights.

1. Put a Transformers-compatible HuBERT model here:
   rvc_models\hubert\config.json
   rvc_models\hubert\pytorch_model.bin

2. Put the RMVPE checkpoint here:
   rvc_models\rmvpe\rmvpe.pt

3. Import an RVC voice folder from the GUI. The folder must contain a .pth
   model and may contain a matching .index and profile.json.

Optional Beatrice v2 backend
-----------------------------
The proprietary Beatrice Runtime and Beatrice model packages are not included.
Select their local folders from the GUI. The application registers external
paths and does not copy or delete those assets.

Advanced path overrides
-----------------------
VOICE_CHANGE_RVC_SOURCE_DIR   compatible RVC source directory
VOICE_CHANGE_RVC_MODELS_DIR   directory containing hubert\ and rmvpe\
VOICE_CHANGE_RVC_DEFAULT_MODEL default model name

Notes
-----
- The executable is unsigned; Windows SmartScreen may display a warning.
- VB-CABLE is optional. Without it, the application starts with the system
  default output and lets you choose another device in the GUI.
- RVC source code included in this package is distributed under its MIT license
  in _internal\licenses\LICENSE.
