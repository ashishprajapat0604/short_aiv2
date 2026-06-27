# fetch_fonts.ps1 — download all 10 caption fonts into .\fonts\
# Run from your project root:  powershell -ExecutionPolicy Bypass -File fetch_fonts.ps1
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path .\fonts | Out-Null

$fonts = @{
  # Hindi / Devanagari
  "NotoSansDevanagari-Bold.ttf" = "https://github.com/notofonts/devanagari/raw/main/fonts/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Bold.ttf"
  "Mukta-Bold.ttf"             = "https://github.com/google/fonts/raw/main/ofl/mukta/Mukta-Bold.ttf"
  "Hind-Bold.ttf"              = "https://github.com/google/fonts/raw/main/ofl/hind/Hind-Bold.ttf"
  "RozhaOne-Regular.ttf"       = "https://github.com/google/fonts/raw/main/ofl/rozhaone/RozhaOne-Regular.ttf"
  "Kalam-Bold.ttf"             = "https://github.com/google/fonts/raw/main/ofl/kalam/Kalam-Bold.ttf"
  # English / Latin
  "Poppins-Bold.ttf"           = "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf"
  "Anton-Regular.ttf"          = "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf"
  "BebasNeue-Regular.ttf"      = "https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf"
  "ArchivoBlack-Regular.ttf"   = "https://github.com/google/fonts/raw/main/ofl/archivoblack/ArchivoBlack-Regular.ttf"
  "FjallaOne-Regular.ttf"      = "https://github.com/google/fonts/raw/main/ofl/fjallaone/FjallaOne-Regular.ttf"
}

foreach ($name in $fonts.Keys) {
  $dest = ".\fonts\$name"
  if (Test-Path $dest) { Write-Host "  skip (exists)  $name"; continue }
  try {
    Invoke-WebRequest -Uri $fonts[$name] -OutFile $dest
    $kb = [math]::Round((Get-Item $dest).Length / 1KB)
    if ($kb -lt 10) { Write-Host "  WARN (tiny)    $name  ${kb}KB — download may have failed" -ForegroundColor Yellow }
    else            { Write-Host "  ok             $name  ${kb}KB" -ForegroundColor Green }
  } catch {
    Write-Host "  FAILED         $name  — $($_.Exception.Message)" -ForegroundColor Red
  }
}
Write-Host "`nDone. Fonts are in .\fonts\  — restart the server to pick them up."
