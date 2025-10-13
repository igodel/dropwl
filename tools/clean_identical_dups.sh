#!/usr/bin/env bash
# Limpia duplicados idénticos (pdf/png/npz) dejando una copia en una ruta canónica.
# Compatible con macOS (Bash 3.2). No usa mapfile ni arrays asociativos.

set -euo pipefail

DUPS_LIST="/tmp/dups_by_name.txt"
KEEP_PDF_PNG_DIR="general_reports"
KEEP_NPZ_DIRS=("data_paper" "data_synth")

if [ ! -f "$DUPS_LIST" ]; then
  echo "[ERR] No existe $DUPS_LIST. Primero genera la lista con el paso de 'duplicados por nombre'."
  exit 1
fi

normalize_ext() {
  # tolower extension
  ext="${1##*.}"
  printf "%s" "$(printf "%s" "$ext" | tr '[:upper:]' '[:lower:]')"
}

for name in $(cat "$DUPS_LIST"); do
  echo "==== $name"

  # Recolectar rutas con ese nombre (NUL-safe)
  files=()
  while IFS= read -r -d '' f; do
    files+=("$f")
  done < <(find . -type f -name "$name" -print0)

  # Si solo hay una copia, nada que hacer
  if [ ${#files[@]} -le 1 ]; then
    echo "[INFO] Solo 1 copia -> skip"
    continue
  fi

  # Calcular hash de la primera y verificar si todos coinciden
  first_hash="$(shasum -a 256 "${files[0]}" | awk '{print $1}')"
  all_equal=1
  for f in "${files[@]}"; do
    h="$(shasum -a 256 "$f" | awk '{print $1}')"
    if [ "$h" != "$first_hash" ]; then
      all_equal=0
      break
    fi
  done

  if [ $all_equal -ne 1 ]; then
    echo "[SKIP] Contenidos distintos con el mismo nombre. Revisión manual requerida."
    for f in "${files[@]}"; do
      echo "   $f"
    done
    continue
  fi

  # Elegir qué copia conservar según heurística (extensión y ruta)
  ext="$(normalize_ext "$name")"
  keep=""

  if [ "$ext" = "pdf" ] || [ "$ext" = "png" ]; then
    # preferimos general_reports/
    for f in "${files[@]}"; do
      case "$f" in
        ./$KEEP_PDF_PNG_DIR/*) keep="$f"; break ;;
      esac
    done
  elif [ "$ext" = "npz" ]; then
    # preferimos data_paper/ o data_synth/
    for f in "${files[@]}"; do
      for d in "${KEEP_NPZ_DIRS[@]}"; do
        case "$f" in
          ./$d/*) keep="$f"; break 2 ;;
        esac
      done
    done
  fi

  # Si no hubo match, conservar la primera
  [ -z "$keep" ] && keep="${files[0]}"

  echo "[KEEP] $keep"
  for f in "${files[@]}"; do
    [ "$f" = "$keep" ] && continue
    echo "[DEL ] $f"
    rm -f "$f"
  done
done

echo "[DONE] Limpieza de duplicados idénticos finalizada."
