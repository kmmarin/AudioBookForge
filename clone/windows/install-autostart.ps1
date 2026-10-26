# Start the clone server when Windows starts, before anyone logs in, and again if
# it stops. It runs as SYSTEM, so no password is stored. Needs an admin PowerShell.
#   Undo: Unregister-ScheduledTask -TaskName "Audiobook Forge clone server"
$ErrorActionPreference = "Stop"
$name = "Audiobook Forge clone server"
$start = Join-Path $PSScriptRoot "start.ps1"
$token = Join-Path (Split-Path $PSScriptRoot -Parent) "server\token.txt"
if (Test-Path $token) { icacls $token /grant:r "SYSTEM:R" | Out-Null }  # save-token.ps1 lets only its user read it
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$start`" -Foreground"
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT1M"  # after the display driver and network are up
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 5 `
  -RestartInterval (New-TimeSpan -Minutes 2) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
"Registered: $name (at startup, as SYSTEM)"
