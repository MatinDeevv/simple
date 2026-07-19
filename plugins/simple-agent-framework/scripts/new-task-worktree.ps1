param([Parameter(Mandatory=$true)][string]$Task, [Parameter(Mandatory=$true)][string]$Branch)
$ErrorActionPreference = 'Stop'
$root=(git rev-parse --show-toplevel).Trim(); if (@(git status --porcelain).Count) { throw 'Refusing worktree creation from dirty checkout.' }
if ($Branch -notmatch '^[a-z0-9][a-z0-9/_-]*$') { throw 'Branch name must be lowercase safe path.' }
if ($Task -notmatch '^[a-z0-9][a-z0-9_-]*$') { throw 'Task name must be a lowercase single path segment.' }
if ([IO.Path]::IsPathRooted($Task) -or $Task.Contains('..') -or $Task.IndexOfAny([IO.Path]::GetInvalidPathChars()) -ge 0) { throw 'Task name is unsafe.' }
$parent=[IO.Path]::GetFullPath((Join-Path (Split-Path $root -Parent) '.simple-worktrees')); New-Item -ItemType Directory -Force -Path $parent | Out-Null
$target=[IO.Path]::GetFullPath((Join-Path $parent $Task)); if (!$target.StartsWith($parent + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Worktree target escapes parent.' }
if (Test-Path $target) { throw "Worktree already exists: $target" }
git show-ref --verify --quiet "refs/heads/$Branch"; if ($LASTEXITCODE -eq 0) { throw "Branch already exists: $Branch" }
git worktree add -b $Branch $target HEAD; if ($LASTEXITCODE -ne 0) { throw "git worktree add failed with exit code $LASTEXITCODE" }
Write-Output $target
