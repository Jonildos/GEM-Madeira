# Módulos de Visão Computacional e OCR — Integração `jade-backend`

Documentação técnica dos dois motores de IA baseados em visão multimodal (Google Gemini) desenvolvidos para o ecossistema **JADE**:

1. **`app/ia/motor_madeira.py`**: Inspeção visual de defeitos em dormentes de madeira (fendas, nós, apodrecimento, empenamento) com estimativa dimensional opcional via régua de 30 cm.
2. **`app/ia/ocr_placa.py`**: Leitura estruturada (OCR) da plaqueta de identificação dos dormentes da CPTM (substituindo a resposta `501` em `POST /ocr/placa`).

---

## 1. Arquitetura e Princípios de Entrega

Ambos os módulos foram projetados como **bibliotecas Python puras (stateless)** para importação direta pelo FastAPI, seguindo restrições reais de infraestrutura do `jade-backend`:

* **100% em Memória (Sem Disco):** Recebem `bytes` diretamente de `UploadFile` ou do Azure Blob Storage e devolvem dicionários Python (`dict`) ou objetos Pydantic direto em RAM. Não criam arquivos temporários, não renomeiam e não apagam fotos (toda persistência continua sendo responsabilidade de `app/storage.py`).
* **Compatível com Azure App Service (Plano F1):** Não carregam pesos pesados de redes neurais locais (PyTorch/EasyOCR/YOLO) na memória RAM do servidor. Todo o processamento pesado ocorre via API externa, mantendo o consumo de memória baixo e seguro para o plano gratuito.
* **Sem Subprocessos / Sem CLI:** Eliminar a chamada via linha de comando evita sobrecarga de processos concorrentes e problemas de disputa de arquivos temporários quando múltiplos inspetores enviam fotos simultaneamente.
* **Contrato `camelCase` e ISO 8601:** Todos os retornos serializam automaticamente para `camelCase` com timestamps em UTC (`ISO 8601`), aderindo às convenções da API do `jade-backend`.

---

## 2. Pré-requisitos e Configuração

### 2.1 Dependências (`requirements.txt`)

Adicione as bibliotecas abaixo ao `requirements.txt` do `jade-backend`:

```text
google-genai>=1.0.0
pydantic>=2.7.0
pillow>=10.3.0
python-dotenv>=1.0.1
```

> **Recomendação para produção:** Após validar na CI, fixe as versões exatas testadas (`==`) para evitar quebras silenciosas em atualizações futuras do SDK `google-genai`.

### 2.2 Variáveis de Ambiente

| Variável | Obrigatória | Descrição |
| :--- | :---: | :--- |
| `GEMINI_API_KEY` | **Sim** | Chave de autenticação do Google AI Studio. |
| `GEMINI_MODEL` | Não | Fixa o modelo utilizado (ex: `gemini-2.5-flash`). Recomendado em produção para manter latência e margem de erro previsíveis. |

* **Desenvolvimento Local:** Adicione as variáveis ao arquivo `.env` na raiz do `jade-backend`.
* **Produção (Azure App Service — `centralus`):** Configure em **Settings > Environment variables** no Portal do Azure. Os módulos leem diretamente do ambiente do sistema operacional sem exigir arquivo `.env` físico.

### 2.3 Estrutura de Pastas Sugerida

```text
jade-backend/
└── app/
    └── ia/
        ├── motor_madeira.py   # Inspeção de defeitos físicos (/inspecoes/{id}/analise-ia)
        └── ocr_placa.py       # Leitura da placa metálica/etiqueta (POST /ocr/placa)
```

---

## 3. Módulo 1: Inspeção de Defeitos em Madeira (`motor_madeira.py`)

### 3.1 Como Funciona (Pipeline Interno)

1. **Validação de Entrada:** Aceita de 1 até **5 imagens** (limite oficial do fluxo de inspeção do JADE) em formato `bytes`, `str` ou `Path`.
2. **Pré-processamento em RAM:** Corrige automaticamente a orientação EXIF (evitando fotos rotacionadas de celulares), converte para RGB e redimensiona proporcionalmente para no máximo **`1024px`** no lado maior (JPEG qualidade 85).
3. **Inferência Estruturada (`temperature=0.1`):** Envia as imagens e suas resoluções exatas em pixels ao Gemini com `response_schema=ResultadoAnaliseIA`, obrigando a resposta a respeitar o contrato Pydantic.
4. **Pós-processamento Determinístico (`@model_validator`):**
   * **Conversão de Escala:** Se uma régua de 30 cm for detectada, o Python calcula o fator `30.0 / comprimento_regua_pixels` e converte as medidas de pixels para centímetros (com arredondamento de 1 casa decimal para evitar falsa precisão metrológica).
   * **Trava Anti-Falso-Positivo (Bug #204):** Se `dormenteDetectado` for `false` ou `qualidadeImagemSuficiente` for `false`, força o veredito para `analise_necessaria`.
   * **Espelho de `app/business_rules.py`:** Calcula `vereditoSugerido` sobre as detecções com confiança $\ge 0.40$:
     * $\ge 1$ detecção `critica` $\rightarrow$ `rejeitado`
     * $\ge 1$ detecção `moderada` ou $\ge 3$ detecções `leve` $\rightarrow$ `quarentena`
     * Caso contrário $\rightarrow$ `aprovado`

### 3.2 Assinatura da Função

```python
from app.ia.motor_madeira import analisar_madeira

resultado = analisar_madeira(
    imagens: list[bytes | str | Path],
    api_key: str | None = None,
    modelo: str | None = None,
    timeout_ms: int = 25000,
    retornar_dict: bool = True,
)
```

### 3.3 Contrato de Retorno (Exemplo em `camelCase`)

```json
{
  "sucesso": true,
  "timestampIso": "2026-11-19T21:30:00+00:00",
  "resolucaoEnviadaPixels": ["1024x768"],
  "modeloUtilizado": "gemini-2.5-flash",
  "tempoProcessamentoSegundos": 2.41,
  "analise": {
    "dormenteDetectado": true,
    "qualidadeImagemSuficiente": true,
    "calibracao": {
      "regua30cmDetectada": true,
      "comprimentoReguaPixels": 310.0,
      "fatorConversaoPxParaCm": 0.09677,
      "comprimentoPecaPixels": 920.0,
      "larguraPecaPixels": 180.0,
      "comprimentoPecaCm": 89.0,
      "larguraPecaCm": 17.4,
      "observacaoEscala": "Régua de 30 cm detectada ao lado da face superior."
    },
    "deteccoes": [
      {
        "tipoDefeito": "fenda_rachadura",
        "severidade": "moderada",
        "confianca": 0.85,
        "localizacao": "topo_extremidade",
        "comprimentoPixels": 150.0,
        "larguraPixels": 6.0,
        "comprimentoCm": 14.5,
        "larguraCm": 0.6,
        "observacao": "Fenda longitudinal partindo do topo da peça."
      }
    ],
    "vereditoSugerido": "quarentena",
    "parecerTecnico": "Dormente apresenta fenda longitudinal moderada na extremidade. Recomenda-se quarentena para verificação de tolerância conforme norma."
  },
  "erro": null
}
```

### 3.4 Limitações Técnicas Conhecidas
* **Natureza das Medidas:** As dimensões em centímetros são **estimativas visuais de ordem de grandeza** sujeitas a distorção de perspectiva (paralaxe). Devem ser exibidas no app como apoio à triagem do inspetor, não como medição metrológica de paquímetro.
* **Empenamento:** Depende de foto tirada pela borda/topo ou com superfície plana de referência.

---

## 4. Módulo 2: OCR de Placa do Dormente CPTM (`ocr_placa.py`)

### 4.1 Como Funciona (Pipeline Interno)

1. **Preservação de Alta Resolução:** Diferente do motor de defeitos, as etiquetas da CPTM ocupam pequena área do quadro. A imagem é mantida em até **`2048px`** (JPEG qualidade 92) com correção EXIF em RAM para preservar fontes pequenas (como nome científico da essência e classe).
2. **Inferência Determinística (`temperature=0.0`):** Temperatura zerada para máxima fidelidade de transcrição.
3. **Foco na Placa Principal:** Quando há pilhas de dormentes no pátio, o prompt instrui o modelo a ler exclusivamente a placa mais próxima/centralizada e ignorar as secundárias.
4. **Regra Anti-Alucinação:** Campos sujos, rasgados ou borrados retornam `null`. Se nenhuma placa for localizada (`placaDetectada=false`), o validador Pydantic zera todos os campos de dados automaticamente, atendendo à política do backend de nunca inventar dados de placa.
5. **Separação de Campos Compostos:** Divide automaticamente rótulos combinados da etiqueta da CPTM:
   * `ESSÊNCIA / CLASSE` $\rightarrow$ `essencia` + `sleeperClass` (`"1a"`, `"2a"`, `"3a"` ou `"especial"`)
   * `TRATAMENTO / EMPRESA` $\rightarrow$ `tratamento` + `empresaTratamento`
   * `EMPRESA INSP. / ANO` $\rightarrow$ `empresaInspecao` + `anoInspecao`
   * `COMPRIMENTO` $\rightarrow$ `comprimentoMetros` (float)

### 4.2 Assinatura da Função

```python
from app.ia.ocr_placa import ler_placa_dormente

resultado = ler_placa_dormente(
    imagem: bytes | str | Path,
    api_key: str | None = None,
    modelo: str | None = None,
    timeout_ms: int = 20000,
    retornar_dict: bool = True,
)
```

### 4.3 Contrato de Retorno (Exemplo em `camelCase`)

Baseado na leitura de uma placa real padrão CPTM:

```json
{
  "sucesso": true,
  "timestampIso": "2026-11-19T21:35:10+00:00",
  "resolucaoEnviadaPixels": "1536x2048",
  "modeloUtilizado": "gemini-2.5-flash",
  "tempoProcessamentoSegundos": 1.85,
  "dados": {
    "placaDetectada": true,
    "confiancaGeral": 0.96,
    "orgao": "CPTM",
    "fornecedor": "TG-TRÊS GURI",
    "ataRegistro": "AR00224",
    "contrato": "AR00224-01",
    "essencia": "DIALIUM GUIANENSE, LEGUMINOSAE",
    "sleeperClass": "1a",
    "comprimentoMetros": 4.0,
    "tratamento": "CCA",
    "empresaTratamento": "GRANOSKI",
    "empresaInspecao": "UFV",
    "anoInspecao": 2024,
    "textoBruto": "CPTM | TG-TRÊS GURI | ATA DE REGISTRO: AR00224 | ESSÊNCIA/CLASSE: DIALIUM GUIANENSE, LEGUMINOSAE / 1a | COMPRIMENTO: 4,00 M | TRATAMENTO/EMPRESA: CCA / GRANOSKI | EMPRESA INSP./ANO: UFV / 2024 | CONTRATO: AR00224-01",
    "observacao": "Todos os campos da placa principal legíveis."
  },
  "erro": null
}
```

---

## 5. Guia de Integração no `jade-backend` (FastAPI)

> **Atenção Crítica de Concorrência (FastAPI + Azure F1):** As chamadas ao SDK `google-genai` são síncronas (I/O bloqueante de 1,5s a 4s). Em rotas `async def`, **sempre** execute os motores via `fastapi.concurrency.run_in_threadpool` (ou declare a rota como `def` comum) para não congelar o *event loop* do servidor para outros usuários.

### 5.1 Integração da Inspeção (`POST /inspecoes/{id}/analise-ia`)

```python
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.ia.motor_madeira import analisar_madeira

router = APIRouter()


@router.post("/inspecoes/{inspecao_id}/analise-ia")
async def executar_analise_ia(
    inspecao_id: str,
    imagens: list[UploadFile] = File(...),
):
    # Respeita o limite de até 5 imagens por inspeção
    imagens_bytes = [await img.read() for img in imagens[:5]]

    resultado = await run_in_threadpool(analisar_madeira, imagens_bytes)

    if not resultado["sucesso"]:
        raise HTTPException(status_code=502, detail=resultado["erro"])

    # Nota: Mapear resultado["analise"] para o schema oficial em app/schemas.py
    # e/ou repassar resultado["analise"]["deteccoes"] para app/business_rules.py
    return resultado
```

### 5.2 Integração do OCR de Placa (`POST /ocr/placa`)

Substitui o retorno atual `501 Not Implemented` e entrega o campo `sleeperClass` pronto para preencher `lotes.classe` no recebimento:

```python
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.ia.ocr_placa import ler_placa_dormente

router = APIRouter()


@router.post("/ocr/placa")
async def executar_ocr_placa(imagem: UploadFile = File(...)):
    imagem_bytes = await imagem.read()

    resultado = await run_in_threadpool(ler_placa_dormente, imagem_bytes)

    if not resultado["sucesso"]:
        raise HTTPException(status_code=502, detail=resultado["erro"])

    if not resultado["dados"]["placaDetectada"]:
        raise HTTPException(
            status_code=422,
            detail="Nenhuma placa legível de dormente foi detectada na imagem.",
        )

    return resultado
```

---

## 6. Testes Locais (Sem Subir o FastAPI)

Para validar ambos os módulos isoladamente antes de plugar nas rotas, execute um script de teste simples passando os bytes das imagens:

```python
import json
from pathlib import Path

from app.ia.motor_madeira import analisar_madeira
from app.ia.ocr_placa import ler_placa_dormente

# 1. Teste de defeitos em madeira
foto_dormente = Path("exemplo_dormente.jpg").read_bytes()
print("=== INSPEÇÃO DE MADEIRA ===")
print(json.dumps(analisar_madeira([foto_dormente]), indent=2, ensure_ascii=False))

# 2. Teste de OCR da placa CPTM
foto_placa = Path("exemplo_placa_cptm.jpg").read_bytes()
print("=== OCR PLACA CPTM ===")
print(json.dumps(ler_placa_dormente(foto_placa), indent=2, ensure_ascii=False))
```

---

## 7. Checklist Antes de Abrir PR (Trava de Contrato / CI)

- [ ] **Trava de Contrato (`contrato/` e `app/schemas.py`):** Conferir e adaptar as chaves de saída dos motores para os schemas exatos esperados pelo front-end, garantindo que `contract.test.ts` e a suíte de testes do backend passem sem quebrar a CI.
- [ ] **Regras Dinâmicas (Tarefa #495):** Conectar a lista `deteccoes` do `motor_madeira.py` às tolerâncias cadastradas na tabela `tipos_defeito` em `app/business_rules.py`, caso o veredito oficial deva ser recalculado dinamicamente por organização.
- [ ] **Prioridade de `sleeperClass`:** No fluxo de recebimento, manter a regra de negócio da wiki onde a classe informada manualmente pelo operador prevalece sobre a classe lida pelo OCR.
- [ ] **UX do App Mobile (Recomendado):** Exibir uma moldura guia na câmera para aproximação da placa na captura de OCR e orientar o uso da régua de 30 cm paralela ao dormente na inspeção de fendas.
- [ ] **Variáveis no Azure:** Confirmar o cadastro de `GEMINI_API_KEY` e `GEMINI_MODEL` no App Service antes do merge na `main`.