# Save the token Forge uses (its CLONE_TOKEN, or SHARE_TOKEN) from stdin, readable
# only by this account, without it appearing on screen or in a command line.
# Printable characters only: a pipe from PowerShell can put a byte-order mark in front.
$token = [Console]::In.ReadToEnd() -replace '[^\x21-\x7E]', ''
if (-not $token) { "No token on stdin"; exit 1 }
$file = Join-Path (Split-Path $PSScriptRoot -Parent) "server\token.txt"
Set-Content -Path $file -Value $token -NoNewline -Encoding ascii
icacls $file /inheritance:r /grant:r "$($env:USERNAME):F" /grant:r "SYSTEM:R" | Out-Null  # SYSTEM: the startup task
"Token saved ($($token.Length) characters)"
