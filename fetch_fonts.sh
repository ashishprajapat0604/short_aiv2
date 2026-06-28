#!/usr/bin/env bash
# fetch_fonts.sh — download all 10 caption fonts into ./fonts/  (WSL / Linux / macOS)
#   bash fetch_fonts.sh
set -u
mkdir -p fonts
declare -A FONTS=(
  # Hindi / Devanagari
  ["NotoSansDevanagari-Bold.ttf"]="https://github.com/notofonts/devanagari/raw/main/fonts/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Bold.ttf"
  ["Mukta-Bold.ttf"]="https://github.com/google/fonts/raw/main/ofl/mukta/Mukta-Bold.ttf"
  ["Hind-Bold.ttf"]="https://github.com/google/fonts/raw/main/ofl/hind/Hind-Bold.ttf"
  ["RozhaOne-Regular.ttf"]="https://github.com/google/fonts/raw/main/ofl/rozhaone/RozhaOne-Regular.ttf"
  ["Kalam-Bold.ttf"]="https://github.com/google/fonts/raw/main/ofl/kalam/Kalam-Bold.ttf"
  ["Baloo2-Bold.ttf"]="https://github.com/google/fonts/raw/main/ofl/baloo2/Baloo2-Bold.ttf"
  ["Laila-Bold.ttf"]="https://github.com/google/fonts/raw/main/ofl/laila/Laila-Bold.ttf"
  ["Rajdhani-Bold.ttf"]="https://github.com/google/fonts/raw/main/ofl/rajdhani/Rajdhani-Bold.ttf"
  # English / Latin
  ["Poppins-Bold.ttf"]="https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf"
  ["Anton-Regular.ttf"]="https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf"
  ["BebasNeue-Regular.ttf"]="https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf"
  ["ArchivoBlack-Regular.ttf"]="https://github.com/google/fonts/raw/main/ofl/archivoblack/ArchivoBlack-Regular.ttf"
  ["FjallaOne-Regular.ttf"]="https://github.com/google/fonts/raw/main/ofl/fjallaone/FjallaOne-Regular.ttf"
  ["Oswald-Bold.ttf"]="https://github.com/google/fonts/raw/main/ofl/oswald/Oswald-Bold.ttf"
  ["Montserrat-Bold.ttf"]="https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Bold.ttf"
  ["Staatliches-Regular.ttf"]="https://github.com/google/fonts/raw/main/ofl/staatliches/Staatliches-Regular.ttf"
  ["BarlowCondensed-Bold.ttf"]="https://github.com/google/fonts/raw/main/ofl/barlowcondensed/BarlowCondensed-Bold.ttf"
  ["Righteous-Regular.ttf"]="https://github.com/google/fonts/raw/main/ofl/righteous/Righteous-Regular.ttf"
)
for name in "${!FONTS[@]}"; do
  dest="fonts/$name"
  if [ -f "$dest" ]; then echo "  skip (exists)  $name"; continue; fi
  if curl -fLs -o "$dest" "${FONTS[$name]}"; then
    kb=$(( $(wc -c < "$dest") / 1024 ))
    if [ "$kb" -lt 10 ]; then echo "  WARN (tiny)    $name  ${kb}KB"; else echo "  ok             $name  ${kb}KB"; fi
  else
    echo "  FAILED         $name"
  fi
done
echo ""
echo "Done. Fonts are in ./fonts/ — restart the server to pick them up."
