# Set up this Windows PC as a converter: Audiobook Forge in WSL 2, on the GPU,
# voicing books (and, if you like, training voices) for an always-on Forge
# server, which keeps the library and serves Read along. In an admin
# PowerShell, from the project folder:
#
#   powershell -ExecutionPolicy Bypass -File deploy\windows-converter.ps1
#
# Run it again to update the app or change the answers; books, settings and
# downloaded models are kept. Step by step it:
#   1. installs WSL 2 with Ubuntu if it's missing (then asks you to restart
#      Windows, open Ubuntu once to make your user, and run it again),
#   2. turns on systemd in Ubuntu (Docker needs it),
#   3. asks for the server to send books to and the token you share with it,
#   4. runs deploy/wsl-deploy.sh, which installs Docker (and NVIDIA's container
#      toolkit, with an NVIDIA GPU) in WSL, then builds and starts Forge,
#   5. with -OpenPort, lets other machines reach this PC's Forge (a port proxy
#      and a firewall rule), so the server can send voice changes back here.
#
# Without an NVIDIA GPU Forge still runs here, on the CPU. An AMD Radeon can
# still voice cloned voices: see clone/windows/ for the clone server.
param(
    [string]$Server,            # the always-on Forge, e.g. http://192.168.1.10:8080 ("none": keep books here)
    [switch]$VoiceCreator,      # train voices here too (needs the NVIDIA GPU)
    [switch]$CloneServer,       # run the clone server here too: cloned voices and pictures (12 GB+ GPU)
    [string]$CloneUrl,          # or use one on another machine, e.g. http://192.168.1.30:8100
    [int]$Port = 8080,
    [string]$Distro = "Ubuntu",
    [switch]$OpenPort
)
$ErrorActionPreference = "Stop"
$env:WSL_UTF8 = "1"  # wsl.exe answers in UTF-8, not UTF-16

function Say($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host "`nError: $text" -ForegroundColor Red; exit 1 }
function InWsl([string]$command) {
    & wsl.exe -d $Distro -u root -- sh -c $command
    return $LASTEXITCODE
}

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { Fail "Run this in a PowerShell opened with Run as administrator." }
$project = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $project "deploy\wsl-deploy.sh"))) { Fail "Run this from the Audiobook Forge folder." }

# 1. WSL 2 and Ubuntu
Say "Checking WSL"
$distros = @()
try { $distros = (& wsl.exe -l -q 2>$null) -replace "`0", "" | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() } } catch {}
if ($distros -notcontains $Distro) {
    Say "Installing WSL 2 with $Distro"
    & wsl.exe --install -d $Distro
    Write-Host @"

WSL is installing. Restart Windows if it asks you to, then open $Distro from the
Start menu once and make your Linux user. After that, run this script again.
"@
    exit 0
}
& wsl.exe --set-version $Distro 2 2>$null | Out-Null

# 2. systemd, for Docker
if ((InWsl "grep -qs '^systemd=true' /etc/wsl.conf") -ne 0) {
    Say "Turning on systemd in $Distro"
    InWsl "printf '\n[boot]\nsystemd=true\n' >> /etc/wsl.conf" | Out-Null
    & wsl.exe --terminate $Distro | Out-Null
}

# 3. The answers
$settings = @()
if (-not $Server) {
    $Server = Read-Host "The always-on Forge server to send finished books to (like http://192.168.1.10:8080; blank: keep them here)"
}
$Server = $Server.Trim().TrimEnd("/")
if ($Server -and $Server -ne "none") {
    if ($Server -notmatch '^https?://') { $Server = "http://$Server" }
    $secure = Read-Host "The server's SHARE_TOKEN (blank: make a new one)" -AsSecureString
    $token = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
    if (-not $token) {
        $bytes = New-Object byte[] 24
        [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $token = -join ($bytes | ForEach-Object { $_.ToString("x2") })
        Write-Host "`nA new token. Put this same line in the server's .env, then restart it there:" -ForegroundColor Yellow
        Write-Host "  SHARE_TOKEN=$token`n"
    }
    # How the server reaches this PC: the address of the network Windows' default route uses.
    $route = Get-NetRoute -DestinationPrefix "0.0.0.0/0" -ErrorAction SilentlyContinue | Sort-Object RouteMetric | Select-Object -First 1
    $ip = $null
    if ($route) {
        $ip = (Get-NetIPAddress -InterfaceIndex $route.InterfaceIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Select-Object -First 1).IPAddress
    }
    $settings += "PUBLISH_URL=$Server", "PUBLISH_AUTO=true", "SHARE_TOKEN=$token"
    if ($ip) { $settings += "PUBLIC_URL=http://${ip}:$Port" }
}
if (-not $PSBoundParameters.ContainsKey("VoiceCreator")) {
    $VoiceCreator = (Read-Host "Train your own voices on this PC too? It needs an NVIDIA GPU (y/N)") -match '^[yY]'
}
$settings += "VOICELAB=$(if ($VoiceCreator) { 'true' } else { 'false' })"
if ($CloneUrl) {
    $settings += "CLONE_URL=$($CloneUrl.TrimEnd('/'))", "CLONE_LOCAL=false"
} else {
    if (-not $PSBoundParameters.ContainsKey("CloneServer")) {
        $CloneServer = (Read-Host ("Run the clone server here too, for cloned voices and Watch and read's pictures? " +
            "It needs an NVIDIA GPU with 12 GB or more (y/N)")) -match '^[yY]'
    }
    $settings += "CLONE_LOCAL=$(if ($CloneServer) { 'true' } else { 'false' })"
    if ($CloneServer) { $settings += "CLONE_URL=http://clone:8100" }  # the clone service beside Forge
}

# 4. Forge itself, in WSL
Say "Installing and starting Audiobook Forge in $Distro (the first time takes a while)"
$wslProject = (& wsl.exe -d $Distro -u root -- wslpath -a ($project -replace '\\', '/')).Trim()
$settings -join "`n" | & wsl.exe -d $Distro -u root -- bash "$wslProject/deploy/wsl-deploy.sh" --port $Port --settings-stdin
if ($LASTEXITCODE -ne 0) { Fail "Setting up Forge in WSL didn't finish (see above). Fix that, then run this again." }

# 5. Reachable from other machines
if ($OpenPort) {
    if (-not $ip) { Fail "Couldn't tell this PC's network address, so the port wasn't opened." }
    Say "Letting other machines reach this PC on port $Port"
    & netsh interface portproxy delete v4tov4 listenaddress=$ip listenport=$Port 2>$null | Out-Null
    & netsh interface portproxy add v4tov4 listenaddress=$ip listenport=$Port connectaddress=127.0.0.1 connectport=$Port | Out-Null
    if (-not (Get-NetFirewallRule -DisplayName "Audiobook Forge" -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName "Audiobook Forge" -Direction Inbound -Protocol TCP -LocalPort $Port `
            -Action Allow -Profile Private, Domain | Out-Null
    }
    Write-Host "Other machines on your network reach it at http://${ip}:$Port. Give this PC a fixed address (a DHCP reservation) so that stays true."
}

Say "Done"
Write-Host "Forge on this PC: http://localhost:$Port"
if ($Server -and $Server -ne "none") {
    Write-Host "Finished books go to $Server. On the server, set the same SHARE_TOKEN in its .env"
    if ($VoiceCreator -and $ip) { Write-Host "and VOICELAB_URL=http://${ip}:$Port to reach the voice creator from there." }
}
