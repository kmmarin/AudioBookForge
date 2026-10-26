# Set this Windows PC up as a GPU server for Audiobook Forge: the clone server,
# which speaks cloned voices and draws Watch and read's pictures for every Forge
# that lists it in Settings → GPU servers. Two ways:
#
#   -Mode windows   natively, with PyTorch for the card: NVIDIA or AMD Radeon.
#                   No WSL or Docker; it starts by itself when Windows does.
#   -Mode wsl       in WSL 2 with Docker: NVIDIA only.
#
# In an admin PowerShell, from the project folder:
#
#   powershell -ExecutionPolicy Bypass -File deploy\gpu-server.ps1 -Mode windows -Token <token>
#
# The token is the one Forge's Settings page shows when you add a server (any
# long random text; the same on both sides). Without -Token one is made and
# printed. Run it again to update; the models it has downloaded are kept.
param(
    [ValidateSet("windows", "wsl")][string]$Mode = "windows",
    [string]$Token,
    [int]$Port = 8100,
    [string]$Root = "C:\forge-clone",   # -Mode windows: where it's installed
    [string]$Distro = "Ubuntu"          # -Mode wsl
)
$ErrorActionPreference = "Stop"
$env:WSL_UTF8 = "1"
function Say($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host "`nError: $text" -ForegroundColor Red; exit 1 }

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { Fail "Run this in a PowerShell opened with Run as administrator." }
$project = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $project "clone\server.py"))) { Fail "Run this from the Audiobook Forge folder." }
if ($Port -ne 8100 -and $Mode -eq "windows") { Fail "The Windows install listens on port 8100." }

$made = $false
if (-not $Token) {
    $bytes = New-Object byte[] 24
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $Token = -join ($bytes | ForEach-Object { $_.ToString("x2") })
    $made = $true
}
$Token = $Token -replace '[^\x21-\x7E]', ''
$route = Get-NetRoute -DestinationPrefix "0.0.0.0/0" -ErrorAction SilentlyContinue | Sort-Object RouteMetric | Select-Object -First 1
$ip = if ($route) { (Get-NetIPAddress -InterfaceIndex $route.InterfaceIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue | Select-Object -First 1).IPAddress }

if ($Mode -eq "windows") {
    Say "Installing the clone server under $Root (Python, PyTorch for the card, the models' code)"
    & (Join-Path $project "clone\windows\setup.ps1") -Root $Root
    if ($LASTEXITCODE) { Fail "The install didn't finish (see above)." }
    Say "Saving the token"
    $Token | & powershell -NoProfile -ExecutionPolicy Bypass -File "$Root\server\save-token.ps1"
    Say "Starting it with Windows"
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$Root\server\stop.ps1" 2>$null | Out-Null  # an older copy goes first
    & "$Root\server\install-autostart.ps1"
    Start-ScheduledTask -TaskName "Audiobook Forge clone server"
} else {
    Say "Checking WSL"
    $distros = @()
    try { $distros = (& wsl.exe -l -q 2>$null) -replace "`0", "" | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() } } catch {}
    if ($distros -notcontains $Distro) {
        & wsl.exe --install -d $Distro
        Write-Host "`nWSL is installing. Restart Windows if it asks, open $Distro once from the Start menu to make your Linux user, then run this again."
        exit 0
    }
    & wsl.exe -d $Distro -u root -- sh -c "grep -qs '^systemd=true' /etc/wsl.conf"
    if ($LASTEXITCODE -ne 0) {
        & wsl.exe -d $Distro -u root -- sh -c "printf '\n[boot]\nsystemd=true\n' >> /etc/wsl.conf"
        & wsl.exe --terminate $Distro | Out-Null
    }
    Say "Installing Docker and the clone server in $Distro"
    $wslProject = (& wsl.exe -d $Distro -u root -- wslpath -a ($project -replace '\\', '/')).Trim()
    $Token | & wsl.exe -d $Distro -u root -- bash "$wslProject/deploy/clone-setup.sh" --from "$wslProject"
    if ($LASTEXITCODE -ne 0) { Fail "Setting it up in WSL didn't finish (see above)." }
    if ($ip) {
        Say "Letting other machines reach it on port $Port"
        & netsh interface portproxy delete v4tov4 listenaddress=$ip listenport=$Port 2>$null | Out-Null
        & netsh interface portproxy add v4tov4 listenaddress=$ip listenport=$Port connectaddress=127.0.0.1 connectport=$Port | Out-Null
    }
}

if (-not (Get-NetFirewallRule -DisplayName "Audiobook Forge GPU server" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName "Audiobook Forge GPU server" -Direction Inbound -Protocol TCP -LocalPort $Port `
        -Action Allow -Profile Private, Domain | Out-Null
}

Say "Waiting for it to answer"
$health = $null
foreach ($i in 1..60) {
    try { $health = Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 3; break } catch { Start-Sleep 3 }
}
if (-not $health) { Fail "It didn't answer on port $Port. Windows: see $Root\server\clone.log. WSL: docker compose -f /opt/audiobook-forge-clone/docker-compose.clone.yml logs" }
Write-Host ("Running on: {0}{1}" -f $(if ($health.gpu) { $health.gpu } else { "the CPU (no GPU found: very slow)" }),
    $(if ($health.gpu_gb) { ", $($health.gpu_gb) GB" }))

Say "Done. In Forge, open Settings → GPU servers → Add a GPU server:"
Write-Host "  Address: http://$(if ($ip) { $ip } else { '<this PC>' }):$Port"
Write-Host "  Token:   $(if ($made) { $Token } else { '(the one you gave)' })"
if ($made) { Write-Host "  (Keep the token: it's also saved on this PC, readable only by this account and the startup task.)" }
Write-Host "Give this PC a fixed address (a DHCP reservation) so Forge keeps finding it."
