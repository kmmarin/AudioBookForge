# Start the clone server in the background (port 8100), logging to clone.log.
# Run from anywhere; it finds its folder. -Foreground keeps it in this window.
param([switch]$Foreground)
$root = Split-Path $PSScriptRoot -Parent
if (Get-NetTCPConnection -LocalPort 8100 -State Listen -ErrorAction SilentlyContinue) { "Already running on port 8100"; exit 0 }
if (-not $Foreground) {
  # A process of its own, so it outlives the window or SSH session that started it.
  $cmd = "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSCommandPath`" -Foreground"
  Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd} | Out-Null
  foreach ($i in 1..30) {
    Start-Sleep 2
    try { Invoke-RestMethod http://127.0.0.1:8100/api/health -TimeoutSec 3 | Out-Null; "Clone server running on port 8100"; exit 0 } catch {}
  }
  "It didn't answer within a minute: see $root\server\clone.log"; exit 1
}
$env:HF_HOME = "$root\hf"
$env:CLONE_DATA = "$root\server-data"
$env:CLONE_IDLE_SECONDS = if ($env:CLONE_IDLE_SECONDS) { $env:CLONE_IDLE_SECONDS } else { "600" }
if (Test-Path "$root\server\token.txt") { $env:CLONE_TOKEN = (Get-Content "$root\server\token.txt" -Raw).Trim() }
Set-Location "$root\server"
& "$root\venv\Scripts\python.exe" -m uvicorn server:app --host 0.0.0.0 --port 8100 *>> "$root\server\clone.log"
