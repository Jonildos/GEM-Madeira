import json
import sys
from pathlib import Path
from analisador_madeira import analisar_madeira

EXTENSOES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

if len(sys.argv) > 1:
    caminhos = [Path(arg) for arg in sys.argv[1:6]]
    for c in caminhos:
        if not c.exists():
            print(f"Arquivo não encontrado: {c}")
            sys.exit(1)
else:
    imagens = [
        p
        for p in Path(__file__).resolve().parent.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSOES
    ]
    if not imagens:
        print("Nenhuma imagem encontrada na pasta do script.")
        sys.exit(1)
    caminhos = [max(imagens, key=lambda p: p.stat().st_mtime)]

print(f"Analisando {len(caminhos)} arquivo(s): {[c.name for c in caminhos]}\n")
imagens_bytes = [c.read_bytes() for c in caminhos]

resultado = analisar_madeira(imagens_bytes)
print(json.dumps(resultado, indent=2, ensure_ascii=False))