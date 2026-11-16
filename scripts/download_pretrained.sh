#!/usr/bin/env bash
# Download the pretrained backbones, split-1 checkpoints and split-1 CSVs from the
# authors' Google Drive folder directly into the places the experiment scripts expect.
# Usage: ./scripts/download_pretrained.sh        (run from the repo root; needs `gdown`)
set -euo pipefail
cd "$(dirname "$0")/.."

command -v gdown >/dev/null || { echo "gdown not found: run 'uv pip install gdown' (or pip install gdown)"; exit 1; }

# path<TAB>drive file id
FILES='
experiments/image/ImageMol.pth	112PqKJG2FRseK3eo-XvDoi5NObBFv4J3
experiments/spectrum/8.pth	1h7H4ZNtrGsPToyVlaFWqXlVWl53dg12r
checkpoints/encoder/train_only/graph_encoder.pth	1FCz36xDprHIXMLNu2BhBC2Ct0PlZEDbN
checkpoints/encoder/train_only/image_encoder.pth	1K973du9FNAR7XlSL78Sekc-hMsOIk8q9
checkpoints/encoder/train_only/smiles_encoder.pth	1ryqZaIn4Jv3VHawTtJbktdm_ROPw5EZn
checkpoints/encoder/train_only/spectrum_encoder.pth	1tAUBj7EkYHlzc1qs1NcT71--y_crz0N6
checkpoints/encoder/train_and_valid/graph_encoder.pth	1UdvtNJ-cnyx8Kzf8zZOZj5lVD9QLsTRh
checkpoints/encoder/train_and_valid/image_encoder.pth	1Y2SCBq9I7yzKVZE9_aAnPZ44AB6XNeTu
checkpoints/encoder/train_and_valid/smiles_encoder.pth	14At75lJBcCdkLnAhN9-vWYkyi3rx3RLs
checkpoints/encoder/train_and_valid/spectrum_encoder.pth	1rGkOjqz9gYmkxKGYc6g0W8gE_Zu-R6d4
checkpoints/model/moltitox.pth	1IaHY2kNTs6XhDok1Y1UgQnsO7vyT7GjW
checkpoints/parameters/graph_best_params.json	1MCdSw8enuDJ8IYK1wnye3FPzD2er4HDs
checkpoints/parameters/smiles_best_params.json	1yQ4GDuTTEmVluwsT1VIcABOfsV-XDSYa
checkpoints/parameters/image_best_params.json	1ki_x4vwD0a6RH_5BQuOPyZytmueHFwgx
checkpoints/parameters/spectrum_best_params.json	1un6JNRfLMEoHRQa3lbKzMWOI9nfsNthS
checkpoints/parameters/moltitox_best_params.json	1sRDRLU9-bMvI59cROokg3ecMMdcXkVLQ
data/train.csv	1L0kW7jnbMUbcRUAYTjalhfEtiCBOsYnV
data/valid.csv	1uk4dh5Dsws7ln39fLqN6TkLqBY_4n4Nv
data/test.csv	1-JZydQCZCBoJW-mjU-p5lLoY-M1q624Y
data/train_spectra.csv	1btLGU9AtgXg_EaFrJw-qbUwqsFs7j0to
data/valid_spectra.csv	11yGSto5kDlU7eyOGuGRsT4io_6Nj8OZq
data/test_spectra.csv	1SKIyEpkabSJ9oajucMHzsb5tg9lZlgH3
'

echo "$FILES" | while IFS=$'\t' read -r path id; do
  [ -z "$path" ] && continue
  if [ -s "$path" ] && [ "${path##*.}" != "csv" ]; then
    echo "skip (exists): $path"; continue
  fi
  mkdir -p "$(dirname "$path")"
  echo "==> $path"
  gdown -O "$path" "$id"
done

echo
echo "Done. Files in place:"
ls -la experiments/image/ImageMol.pth experiments/spectrum/8.pth checkpoints/model checkpoints/parameters checkpoints/encoder/*
