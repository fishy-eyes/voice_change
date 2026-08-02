param(
    [string]$PythonExecutable = "python",
    [Parameter(Mandatory = $true)]
    [string]$RvcSourceDir
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedRvcSource = (Resolve-Path -LiteralPath $RvcSourceDir).Path
$licensePath = Join-Path $resolvedRvcSource "LICENSE"
if (-not (Test-Path -LiteralPath $licensePath -PathType Leaf)) {
    throw "RVC source LICENSE not found: $licensePath"
}

Push-Location $repoRoot
try {
    $env:VOICE_CHANGE_BUILD_RVC_SOURCE_DIR = $resolvedRvcSource
    & $PythonExecutable -m PyInstaller --noconfirm --clean packaging\voice_change.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }

    $appDirectory = Join-Path $repoRoot "dist\VoiceChanger-v1.0.0"
    $hubertDirectory = Join-Path $appDirectory "rvc_models\hubert"
    $rmvpeDirectory = Join-Path $appDirectory "rvc_models\rmvpe"
    $voiceDirectory = Join-Path $appDirectory "models\rvc"
    New-Item -ItemType Directory -Force $hubertDirectory | Out-Null
    New-Item -ItemType Directory -Force $rmvpeDirectory | Out-Null
    New-Item -ItemType Directory -Force $voiceDirectory | Out-Null
    Copy-Item packaging\README-Windows.txt (Join-Path $appDirectory "README.txt") -Force
    Copy-Item packaging\HuBERT-README.txt (Join-Path $hubertDirectory "README.txt") -Force
    Copy-Item packaging\RMVPE-README.txt (Join-Path $rmvpeDirectory "README.txt") -Force
    Copy-Item packaging\VoiceModels-README.txt (Join-Path $voiceDirectory "README.txt") -Force

    $archive = Join-Path $repoRoot "dist\voice_change-v1.0.0-windows-x64.tar.xz"
    if (Test-Path -LiteralPath $archive) {
        Remove-Item -LiteralPath $archive -Force
    }
    & tar.exe -cJf $archive -C (Split-Path -Parent $appDirectory) (Split-Path -Leaf $appDirectory)
    if ($LASTEXITCODE -ne 0) {
        throw "tar.xz packaging failed with exit code $LASTEXITCODE"
    }
    $hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    $hashFile = "$archive.sha256"
    [System.IO.File]::WriteAllText(
        $hashFile,
        "$hash  $(Split-Path -Leaf $archive)`r`n",
        [System.Text.UTF8Encoding]::new($false)
    )
    Write-Output "Archive: $archive"
    Write-Output "SHA256: $hash"
}
finally {
    Remove-Item Env:VOICE_CHANGE_BUILD_RVC_SOURCE_DIR -ErrorAction SilentlyContinue
    Pop-Location
}
