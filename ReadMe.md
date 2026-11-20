# Motor de Inspeção de Madeira (Gemini Vision) — Integração `jade-backend`

Módulo Python puro para análise visual de dormentes de madeira (fendas, nós, empenamento e estimativa dimensional via régua opcional de 30 cm), projetado para substituir o motor simulado em `app/ia/` no `jade-backend`.

## Características de Produção
- **100% em memória (Stateless):** Recebe `list[bytes]` diretamente do FastAPI (`UploadFile`) ou do Azure Blob Storage. Não grava, renomeia nem apaga arquivos no disco do servidor.
- **Compatível com Azure App Service (Plano F1):** Redimensiona imagens em RAM para no máximo `1024px` antes do envio, economizando memória e banda.
- **Contrato `camelCase`:** Todos os retornos seguem a convenção de nomenclatura do `jade-backend` e datas em ISO 8601 (UTC).
- **Alinhado a `app/business_rules.py`:** Retorna lista de detecções individuais com `confianca` (0.0 a 1.0) e trava `dormenteDetectado` para evitar falso `aprovado` (Bug #204).

---

## 1. Instalação de Dependências

Adicione ao `requirements.txt` do `jade-backend`:

```text
google-genai>=1.0.0
pydantic>=2.7.0
pillow>=10.3.0
python-dotenv>=1.0.1