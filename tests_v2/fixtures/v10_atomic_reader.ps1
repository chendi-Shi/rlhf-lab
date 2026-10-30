param(
    [Parameter(Mandatory=$true)][string]$Target,
    [Parameter(Mandatory=$true)][string]$Ready,
    [Parameter(Mandatory=$true)][string]$SeenNew,
    [Parameter(Mandatory=$true)][string]$Done,
    [Parameter(Mandatory=$true)][string]$Result,
    [Parameter(Mandatory=$true)][string]$OldSha256,
    [Parameter(Mandatory=$true)][string]$NewSha256
)
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSVersion.Major -lt 7) { throw 'PowerShell 7 required' }
$counts = @{ old = 0; new = 0 }
$errors = [Collections.Generic.List[string]]::new()
$timer = [Diagnostics.Stopwatch]::StartNew()
try {
    while (-not [IO.File]::Exists($Done)) {
        if ($timer.Elapsed.TotalSeconds -ge 10) { throw 'Synthetic reader fixture deadline' }
        $stream = [IO.File]::Open($Target, [IO.FileMode]::Open, [IO.FileAccess]::Read,
            [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete)
        $buffer = [IO.MemoryStream]::new()
        try { $stream.CopyTo($buffer); $bytes = $buffer.ToArray() }
        finally { $buffer.Dispose(); $stream.Dispose() }
        # Hash and parse exactly the same complete bytes after closing handles.
        $sha = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
        $value = [Text.Encoding]::UTF8.GetString($bytes) | ConvertFrom-Json -DateKind String
        if ($value.complete -ne $true) { throw 'Partial synthetic status object' }
        if ($sha -ceq $OldSha256) { $counts.old += 1 }
        elseif ($sha -ceq $NewSha256) {
            $counts.new += 1
            if (-not [IO.File]::Exists($SeenNew)) { [IO.File]::WriteAllText($SeenNew, 'seen') }
        } else { throw "Unexpected complete-byte hash: $sha" }
        if (-not [IO.File]::Exists($Ready)) { [IO.File]::WriteAllText($Ready, 'ready') }
        Start-Sleep -Milliseconds 2
    }
} catch { $errors.Add($_.Exception.ToString()) }
[ordered]@{ kind='alignlab-v10-synthetic-ps7-reader'; counts=$counts;
    errors=@($errors); ps_version=$PSVersionTable.PSVersion.ToString();
    model_loaded=$false; torch_loaded=$false; cuda_loaded=$false } |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Result -Encoding utf8
if ($errors.Count -gt 0) { exit 1 }
