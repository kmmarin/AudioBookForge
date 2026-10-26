# The clone server on Windows itself, with no WSL or Docker: PyTorch for the
# card, NVIDIA (CUDA) or AMD Radeon (AMD's ROCm builds for Windows; ROCm in WSL
# doesn't support older Radeons such as the RX 6000 series). Everything goes
# under one folder:
#
#   powershell -ExecutionPolicy Bypass -File setup.ps1 [-Root C:\forge-clone] [-Gpu auto|nvidia|amd] [-Gfx gfx103X-dgpu]
#
# deploy\gpu-server.ps1 runs this, saves the token and starts it at boot. By hand:
# start.ps1 runs it on port 8100; the token Forge uses goes in <Root>\server\token.txt
# (save-token.ps1 reads it from stdin).
param([string]$Root = "C:\forge-clone", [ValidateSet("auto", "nvidia", "amd")][string]$Gpu = "auto",
      [string]$Gfx = "", [string]$Python = "3.13")
$ErrorActionPreference = "Stop"

# Which card: the first NVIDIA or AMD one Windows reports.
$cards = (Get-CimInstance Win32_VideoController).Name
if ($Gpu -eq "auto") {
  if ($cards -match "NVIDIA") { $Gpu = "nvidia" }
  elseif ($cards -match "AMD|Radeon") { $Gpu = "amd" }
  else { throw "No NVIDIA or AMD graphics card found ($($cards -join ', ')). Pass -Gpu nvidia or -Gpu amd." }
}
if ($Gpu -eq "amd" -and -not $Gfx) {
  # AMD's nightly wheels are per GPU family (rocm.nightlies.amd.com/v2/).
  $radeon = $cards | Where-Object { $_ -match "Radeon" } | Select-Object -First 1
  $Gfx = switch -Regex ($radeon) {
    "RX\s*5\d{3}" { "gfx101X-dgpu" }
    "RX\s*6\d{3}" { "gfx103X-dgpu" }
    "RX\s*7\d{3}" { "gfx110X-dgpu" }
    "RX\s*9\d{3}" { "gfx120X-all" }
    default { throw "Couldn't tell the Radeon's family from '$radeon'. Pass -Gfx, like gfx110X-dgpu (see rocm.nightlies.amd.com/v2/)." }
  }
}
"GPU: $Gpu$(if ($Gfx) { " ($Gfx)" })"

# Python, through the py launcher; winget installs it when it's missing.
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { throw "Install Python $Python from python.org (with the py launcher), then run this again." }
  winget install -e --id "Python.Python.$Python" --scope machine --accept-source-agreements --accept-package-agreements
  $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}
New-Item -ItemType Directory -Force "$Root\server", "$Root\hf", "$Root\server-data" | Out-Null
if (-not (Test-Path "$Root\venv\Scripts\python.exe")) { py "-$Python" -m venv "$Root\venv" }
$py = "$Root\venv\Scripts\python.exe"
& $py -m pip install -q --upgrade pip
& $py -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
  if ($Gpu -eq "nvidia") {
    # CUDA 12.8 builds: RTX 20xx through 50xx, and GTX 16xx.
    & $py -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
    if ($LASTEXITCODE -ne 0) { throw "Installing PyTorch for NVIDIA failed (see above)" }
  } else {
    # The staging index is the one that had Windows wheels for gfx103X in October 2026.
    $done = $false
    foreach ($idx in "https://rocm.nightlies.amd.com/v2/$Gfx/", "https://rocm.nightlies.amd.com/v2-staging/$Gfx/") {
      & $py -m pip install --index-url $idx torch torchaudio
      if ($LASTEXITCODE -eq 0) { $done = $true; break }
    }
    if (-not $done) { throw "No ROCm PyTorch for $Gfx on Windows from AMD's indexes" }
  }
}
# qwen-tts without its own dependency list: it would pull another PyTorch over this one.
& $py -m pip install -q --no-deps qwen-tts==0.1.1
& $py -m pip install -q transformers==4.57.3 accelerate==1.12.0 librosa soundfile einops sox onnxruntime `
    huggingface_hub fastapi==0.162.0 uvicorn==0.54.0 diffusers==0.37.1  # diffusers: Watch and read's pictures
Copy-Item -Force "$PSScriptRoot\..\server.py" "$Root\server\server.py"
Copy-Item -Force "$PSScriptRoot\start.ps1", "$PSScriptRoot\stop.ps1", "$PSScriptRoot\save-token.ps1", `
  "$PSScriptRoot\install-autostart.ps1" "$Root\server\"
& $py -c "import torch; ok = torch.cuda.is_available(); print('PyTorch sees:', torch.cuda.get_device_name(0) if ok else 'no GPU (CPU only)')"
