# Stop the clone server: let go of the model first (stopping a process in the
# middle of GPU work can leave AMD's driver stuck until a restart), then end it.
$conn = Get-NetTCPConnection -LocalPort 8100 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $conn) { "Not running"; exit 0 }
$token = $null
$file = Join-Path (Split-Path $PSScriptRoot -Parent) "server\token.txt"
if (Test-Path $file) { $token = (Get-Content $file -Raw).Trim() }
try { Invoke-RestMethod -Method Post http://127.0.0.1:8100/api/unload -Headers @{"X-Forge-Token" = $token} -TimeoutSec 600 | Out-Null } catch {}
Stop-Process -Id $conn.OwningProcess
"Stopped"
