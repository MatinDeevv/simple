param([Parameter(Mandatory=$true)][string]$Task, [Parameter(Mandatory=$true)][string]$Branch)
$ErrorActionPreference = 'Stop'
$root=(git rev-parse --show-toplevel).Trim(); if (@(git status --porcelain).Count) { throw 'Refusing worktree creation from dirty checkout.' }
if ($Branch -notmatch '^[a-z0-9][a-z0-9/_-]*$') { throw 'Branch name must be lowercase safe path.' }
$parent=Join-Path (Split-Path $root -Parent) '.simple-worktrees'; New-Item -ItemType Directory -Force -Path $parent | Out-Null
$target=Join-Path $parent $Task; if (Test-Path $target) { throw "Worktree already exists: $target" }
git worktree add -b $Branch $target HEAD; Write-Output $target
