# The clone server on Windows itself, for a GPU that WSL and Docker can't use:
# an AMD Radeon, through AMD's ROCm builds of PyTorch for Windows. (An RX 6800
# isn't supported by ROCm in WSL.) Everything goes under one folder:
#
#   powershell -ExecutionPolicy Bypass -File setup.ps1 [-Root D:\forge-clone] [-Gfx gfx103X-dgpu]
#
# Then start.ps1 runs it on port 8100. The token Forge uses goes in
# <Root>\server\token.txt (save-token.ps1 reads it from stdin).
param([string]$Root = "D:\forge-clone", [string]$Gfx = "gfx103X-dgpu", [string]$Python = "3.13")
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force "$Root\server", "$Root\hf", "$Root\server-data" | Out-Null
if (-not (Test-Path "$Root\venv\Scripts\python.exe")) { py "-$Python" -m venv "$Root\venv" }
$py = "$Root\venv\Scripts\python.exe"
& $py -m pip install -q --upgrade pip
& $py -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
  # AMD's nightly ROCm wheels for this GPU family (gfx103X: RX 6000 series). The
  # staging index is the one that had Windows wheels for it in October 2026.
  $done = $false
  foreach ($idx in "https://rocm.nightlies.amd.com/v2/$Gfx/", "https://rocm.nightlies.amd.com/v2-staging/$Gfx/") {
    & $py -m pip install --index-url $idx torch torchaudio
    if ($LASTEXITCODE -eq 0) { $done = $true; break }
  }
  if (-not $done) { throw "No ROCm PyTorch for $Gfx on Windows from AMD's indexes" }
}
# qwen-tts without its own dependency list: it would pull CUDA PyTorch over the ROCm one.
& $py -m pip install -q --no-deps qwen-tts==0.1.1
& $py -m pip install -q transformers==4.57.3 accelerate==1.12.0 librosa soundfile einops sox onnxruntime `
    huggingface_hub fastapi==0.162.0 uvicorn==0.54.0 diffusers==0.37.1  # diffusers: Watch and read's pictures
Copy-Item -Force "$PSScriptRoot\..\server.py" "$Root\server\server.py"
Copy-Item -Force "$PSScriptRoot\start.ps1", "$PSScriptRoot\stop.ps1", "$PSScriptRoot\save-token.ps1" "$Root\server\"
& $py -c "import torch; ok = torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0) if ok else 'none (CPU only)')"
