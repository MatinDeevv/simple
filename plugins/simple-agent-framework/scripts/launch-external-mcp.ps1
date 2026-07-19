param([Parameter(Mandatory=$true)][ValidateSet('perplexity','firecrawl')][string]$Server)
$ErrorActionPreference = 'Stop'
$envPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\.env'))
if (!(Test-Path $envPath -PathType Leaf)) { [Console]::Error.WriteLine("$Server MCP: .env missing"); exit 2 }
$keyName = if ($Server -eq 'perplexity') { 'PERPLEXITY_API_KEY' } else { 'FIRECRAWL_API_KEY' }
$line = Get-Content $envPath | Where-Object { $_ -match "^$keyName=(.*)$" } | Select-Object -First 1
if (!$line) { [Console]::Error.WriteLine("$Server MCP: $keyName missing from .env"); exit 2 }
$value = ($line -split '=',2)[1].Trim().Trim('"').Trim("'")
if (!$value) { [Console]::Error.WriteLine("$Server MCP: $keyName empty"); exit 2 }
Set-Item "Env:$keyName" $value
if ($Server -eq 'perplexity') { & npx -y @perplexity-ai/mcp-server } else { & npx -y firecrawl-mcp }
exit $LASTEXITCODE
