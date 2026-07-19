$ErrorActionPreference = 'Stop'
$root = (git rev-parse --show-toplevel).Trim(); $branch = (git branch --show-current).Trim(); $sha = (git rev-parse HEAD).Trim(); $dirty = @(git status --porcelain)
$receipt = Join-Path $root '.agents\receipts'; New-Item -ItemType Directory -Force -Path $receipt | Out-Null
[ordered]@{utc=[DateTime]::UtcNow.ToString('o');root=$root;branch=$branch;head=$sha;dirty_count=$dirty.Count} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $receipt 'preflight.json')
if ($dirty.Count) { Write-Warning 'Dirty worktree: do not alter files without ownership confirmation.' }; Write-Output "preflight: $branch @ $sha"
