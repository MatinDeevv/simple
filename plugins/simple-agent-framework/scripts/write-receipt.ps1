param([Parameter(Mandatory=$true)][string]$Command, [Parameter(Mandatory=$true)][int]$ExitCode, [Parameter(Mandatory=$true)][string]$LogPath)
$ErrorActionPreference='Stop'; if (!(Test-Path $LogPath -PathType Leaf)) { throw "Log missing: $LogPath" }
$root=(git rev-parse --show-toplevel).Trim(); $dir=Join-Path $root '.agents\receipts'; New-Item -ItemType Directory -Force -Path $dir | Out-Null
$hash=(Get-FileHash $LogPath -Algorithm SHA256).Hash.ToLower(); $id=[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
[ordered]@{utc=[DateTime]::UtcNow.ToString('o');head=(git rev-parse HEAD).Trim();command=$Command;exit_code=$ExitCode;log_path=$LogPath;log_sha256=$hash;python=(python --version 2>&1)} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $dir "test-$id.json")
exit $ExitCode
