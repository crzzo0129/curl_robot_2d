param(
    [switch]$Apply
)

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

function Assert-InProject([string]$Path) {
    $resolved = [System.IO.Path]::GetFullPath($Path)
    if (-not $resolved.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to touch a path outside the project: $resolved"
    }
    return $resolved
}

$targets = @()
$targets += Get-ChildItem -LiteralPath $projectRoot -Recurse -Force -Directory |
    Where-Object { $_.Name -in @("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache") }

$pptResult = Join-Path $projectRoot "results/design_logic_exploration_ppt"
if (Test-Path -LiteralPath $pptResult) {
    $targets += Get-ChildItem -LiteralPath $pptResult -Force -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -in @(
                "pptx_check",
                "pptx_check.zip",
                "CURL_robot_design_exploration_CN",
                "CURL_robot_design_exploration_CN.pptx.inspect.ndjson",
                "montage.webp",
                "contact_sheet.png",
                "final_render_montage.png"
            ) -or $_.Name -like "slide-*.png" -or $_.Name -like "slide-*.layout.json"
        }
}

$targets = $targets | Sort-Object FullName -Unique

if (-not $targets) {
    Write-Host "No generated cache or report-build artifacts found."
    exit 0
}

foreach ($target in $targets) {
    $safePath = Assert-InProject $target.FullName
    if ($Apply) {
        Remove-Item -LiteralPath $safePath -Recurse -Force
        Write-Host "Removed: $safePath"
    }
    else {
        Write-Host "Would remove: $safePath"
    }
}

if (-not $Apply) {
    Write-Host "Dry run only. Re-run with -Apply to remove the listed artifacts."
}
