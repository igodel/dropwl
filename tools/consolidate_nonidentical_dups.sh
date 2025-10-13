#!/usr/bin/env bash
# Consolida duplicados NO idénticos (mismo nombre, distinto contenido)
# Copia cada variante a general_reports con nombre único y genera un índice CSV.
# Compatible con macOS (Bash 3.2)

set -euo pipefail

DUPS_LIST="/tmp/dups_by_name.txt"
OUTDIR="general_reports"
INDEX_CSV="$OUTDIR/_index_nonidentical_dups.csv"

[ -d "$OUTDIR" ] || mkdir -p "$OUTDIR"

if [ ! -f "$DUPS_LIST" ]; then
  echo "[ERR] Falta $DUPS_LIST. Genera la lista con el comando de 'uniq -d'." >&2
  exit 1
fi

# Encabezado del índice
echo "old_path,new_path,sha256" > "$INDEX_CSV"

normalize_ext() {
  ext="${1##*.}"
  printf "%s" "$(printf "%s" "$ext" | tr '[:upper:]' '[:lower:]')"
}

sanitize() {
  # Convierte ruta relativa en token seguro para el nombre de archivo
  # Reemplaza '/' y espacios, y filtra caracteres raros.
  printf "%s" "$1" \
  | sed -e 's#^\./##' \
        -e 's#[/ ]#_#g' \
  | tr -cd 'A-Za-z0-9_.-'
}

while read -r name; do
  # Recolectar TODAS las rutas existentes con ese nombre
  paths=()
  while IFS= read -r -d '' f; do
    paths+=("$f")
  done < <(find . -type f -name "$name" -print0)

  # Si por error hay <=1, saltar
  [ ${#paths[@]} -le 1 ] && continue

  # Para cada variante, calcular hash y copiar con nombre único
  for p in "${paths[@]}"; do
    sha="$(shasum -a 256 "$p" | awk '{print $1}')"
    ext="$(normalize_ext "$name")"
    base="${name%.*}"
    tag="$(sanitize "$p")"
    new="${OUTDIR}/${base}__${tag}__${sha:0:8}.${ext}"

    if [ ! -f "$new" ]; then
      cp -p "$p" "$new"
      echo "[COPY] $p -> $new"
    else
      echo "[SKIP] Ya existía: $new"
    fi
    # Añadir al índice
    # Escapar comas en CSV (muy raro en rutas, pero por si acaso)
    old_csv="$(printf "%s" "$p"   | sed 's/,/_/g')"
    new_csv="$(printf "%s" "$new" | sed 's/,/_/g')"
    echo "$old_csv,$new_csv,$sha" >> "$INDEX_CSV"
  done
done < "$DUPS_LIST"

echo "[DONE] Copia consolidada en $OUTDIR"
echo "[INFO] Índice: $INDEX_CSV"
