param(
    [Parameter(Mandatory = $true)]
    [string]$ArchivePath,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

if (-not (Test-Path $ArchivePath)) {
    throw "Archive not found: $ArchivePath"
}

$archiveFullPath = (Resolve-Path $ArchivePath).Path
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$tempRoot = Join-Path $root "runtime\_state_import_$timestamp"

if (Test-Path $tempRoot) {
    Remove-Item $tempRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null

$copied = 0
$skipped = 0
$ignored = 0

try {
    Expand-Archive -Path $archiveFullPath -DestinationPath $tempRoot -Force

    $manifestPath = Join-Path $tempRoot "manifest.json"
    if (Test-Path $manifestPath) {
        try {
            $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
            Write-Host "Import manifest: exported_at=$($manifest.exported_at) machine=$($manifest.machine)"
        }
        catch {
            Write-Host "Manifest present but unreadable; continuing import."
        }
    }

    $files = Get-ChildItem -Path $tempRoot -Recurse -File
    foreach ($file in $files) {
        $relative = $file.FullName.Substring($tempRoot.Length).TrimStart('\\', '/')
        if ($relative -eq "manifest.json") {
            continue
        }

        $normalized = $relative -replace '/', '\\'
        if (-not ($normalized.StartsWith("data\\") -or $normalized.StartsWith("runtime\\"))) {
            $ignored++
            continue
        }

        $destination = Join-Path $root $normalized
        $destDir = Split-Path -Parent $destination
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }

        if ((Test-Path $destination) -and -not $Overwrite) {
            $skipped++
            continue
        }

        Copy-Item -Path $file.FullName -Destination $destination -Force
        $copied++
    }

    Write-Host "State import complete. copied=$copied skipped_existing=$skipped ignored_non_state=$ignored"
    if (-not $Overwrite) {
        Write-Host "Tip: re-run with -Overwrite to replace existing state files."
    }
}
finally {
    if (Test-Path $tempRoot) {
        Remove-Item $tempRoot -Recurse -Force
    }
}
