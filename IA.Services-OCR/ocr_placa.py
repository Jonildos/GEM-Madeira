import io
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator

load_dotenv(override=False)

# Cache em memória para não consultar client.models.list() em toda requisição do FastAPI
_CACHE_MODELOS_DISPONIVEIS: Optional[list[str]] = None

MODELOS_PREFERIDOS = [
    "gemini-3.1-flash",
    "gemini-3.0-flash",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

TERMOS_IGNORADOS = (
    "tts",
    "image",
    "embedding",
    "veo",
    "imagen",
    "audio",
    "robotics",
    "gemma",
    "aqa",
    "live",
)


def _to_camel(string: str) -> str:
    partes = string.split("_")
    return partes[0] + "".join(p.capitalize() for p in partes[1:])


class BaseCamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
    )


class DadosPlacaCPTM(BaseCamelModel):
    placa_detectada: bool = Field(
        description="True se houver uma placa/etiqueta de identificação de dormente visível e legível."
    )
    confianca_geral: float = Field(
        ge=0.0,
        le=1.0,
        description="Confiança média da leitura dos campos (0.0 a 1.0).",
    )
    orgao: Optional[str] = Field(None, description="Ex: CPTM")
    fornecedor: Optional[str] = Field(None, description="Ex: TG-TRÊS GURI")
    ata_registro: Optional[str] = Field(None, description="Ex: AR00224")
    contrato: Optional[str] = Field(None, description="Ex: AR00224-01")
    essencia: Optional[str] = Field(
        None,
        description="Nome científico/espécie da madeira. Ex: DIALIUM GUIANENSE, LEGUMINOSAE",
    )
    sleeper_class: Optional[Literal["1a", "2a", "3a", "especial"]] = Field(
        None,
        description="Classe do dormente extraída do campo ESSÊNCIA / CLASSE (padronizada: '1a', '2a', '3a' ou 'especial'). Null se ilegível.",
    )
    comprimento_metros: Optional[float] = Field(
        None, ge=0, description="Comprimento numérico em metros. Ex: 4.0"
    )
    tratamento: Optional[str] = Field(
        None, description="Tipo de tratamento preservativo. Ex: CCA"
    )
    empresa_tratamento: Optional[str] = Field(
        None, description="Empresa responsável pelo tratamento. Ex: GRANOSKI"
    )
    empresa_inspecao: Optional[str] = Field(
        None, description="Empresa/entidade inspetora. Ex: UFV"
    )
    ano_inspecao: Optional[int] = Field(
        None, ge=1990, le=2100, description="Ano de inspeção/fabricação. Ex: 2024"
    )
    texto_bruto: Optional[str] = Field(
        None, description="Transcrição literal de todo o texto legível na placa principal."
    )
    observacao: str = Field(
        description="Aviso curto caso algum campo esteja borrado, cortado, sujo ou ilegível."
    )

    @model_validator(mode="after")
    def validar_leitura(self) -> "DadosPlacaCPTM":
        # Regra anti-alucinação: se não detectou placa, garante que nenhum dado fique preenchido
        if not self.placa_detectada:
            self.confianca_geral = 0.0
            self.orgao = None
            self.fornecedor = None
            self.ata_registro = None
            self.contrato = None
            self.essencia = None
            self.sleeper_class = None
            self.comprimento_metros = None
            self.tratamento = None
            self.empresa_tratamento = None
            self.empresa_inspecao = None
            self.ano_inspecao = None
        return self


class RespostaOcrPlaca(BaseCamelModel):
    sucesso: bool
    timestamp_iso: str
    resolucao_enviada_pixels: str
    modelo_utilizado: Optional[str] = None
    tempo_processamento_segundos: float
    dados: Optional[DadosPlacaCPTM] = None
    erro: Optional[str] = None


SYSTEM_PROMPT_OCR = """
Você é um sistema especializado em OCR de placas de identificação de dormentes ferroviários da CPTM.
Sua tarefa é extrair com precisão rigorosa os dados impressos na placa/etiqueta.

Regras obrigatórias:
1. Foco na Placa Principal: Se houver vários dormentes empilhados na foto, leia EXCLUSIVAMENTE a placa em primeiro plano / mais centralizada e nítida. Ignore as placas secundárias ao fundo ou abaixo.
2. Zero Alucinação: Se um campo estiver borrado, sujo, coberto por prego ou ilegível, retorne `null` nesse campo. NUNCA invente ou adivinhe códigos, números de contrato ou datas.
3. Separação de Campos Compostos:
   - `ESSÊNCIA / CLASSE` -> separe a espécie em `essencia` e a classe em `sleeper_class` ('1a', '2a', '3a' ou 'especial').
   - `TRATAMENTO / EMPRESA` -> separe em `tratamento` e `empresa_tratamento`.
   - `EMPRESA INSP. / ANO` -> separe em `empresa_inspecao` e `ano_inspecao` (inteiro).
   - `COMPRIMENTO` -> converta para número decimal em metros em `comprimento_metros` (ex: '4,00 M' -> 4.0).
4. Sem Placa: Se a foto não contiver placa de identificação de dormente, defina `placa_detectada=false`.
"""


def _preparar_imagem_ocr(
    entrada: Union[bytes, str, Path],
    max_lado: int = 2048,
) -> tuple[types.Part, tuple[int, int]]:
    """Mantém alta resolução (até 2048px) para preservar textos pequenos da etiqueta."""
    fonte = io.BytesIO(entrada) if isinstance(entrada, (bytes, bytearray)) else entrada

    with Image.open(fonte) as img:
        img_corrigida = ImageOps.exif_transpose(img).convert("RGB")
        img_corrigida.thumbnail((max_lado, max_lado), Image.Resampling.LANCZOS)
        dimensoes = img_corrigida.size

        buffer = io.BytesIO()
        img_corrigida.save(buffer, format="JPEG", quality=92)
        bytes_otimizados = buffer.getvalue()

    part = types.Part.from_bytes(data=bytes_otimizados, mime_type="image/jpeg")
    return part, dimensoes


def _obter_modelos_cascata(
    client: genai.Client, modelo_forcado: Optional[str]
) -> list[str]:
    """Monta fila de fallback priorizando modelos ativos na conta, com cache em RAM."""
    global _CACHE_MODELOS_DISPONIVEIS

    if modelo_forcado:
        return [modelo_forcado]

    env_model = os.getenv("GEMINI_MODEL", "").strip()
    if env_model:
        return [env_model]

    if _CACHE_MODELOS_DISPONIVEIS is None:
        disponiveis = []
        try:
            for m in client.models.list():
                nome = m.name.replace("models/", "")
                if nome.startswith("gemini-") and not any(
                    t in nome for t in TERMOS_IGNORADOS
                ):
                    disponiveis.append(nome)
        except Exception:
            disponiveis = []

        # Coloca os preferidos no topo (se existirem na conta) e o restante logo atrás
        fila = [m for m in MODELOS_PREFERIDOS if m in disponiveis] + [
            m for m in disponiveis if m not in MODELOS_PREFERIDOS
        ]
        _CACHE_MODELOS_DISPONIVEIS = fila if fila else MODELOS_PREFERIDOS

    return _CACHE_MODELOS_DISPONIVEIS


def ler_placa_dormente(
    imagem: Union[bytes, str, Path],
    api_key: Optional[str] = None,
    modelo: Optional[str] = None,
    timeout_ms: int = 20000,
    retornar_dict: bool = True,
) -> Union[dict, RespostaOcrPlaca]:
    """
    Executa OCR estruturado na placa metálica/etiqueta de um dormente da CPTM.
    """
    inicio = time.time()
    resolucao_str = ""

    try:
        chave = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not chave:
            raise PermissionError("Variável de ambiente GEMINI_API_KEY não configurada.")

        part, (largura, altura) = _preparar_imagem_ocr(imagem)
        resolucao_str = f"{largura}x{altura}"

        client = genai.Client(
            api_key=chave,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_OCR,
            response_mime_type="application/json",
            response_schema=DadosPlacaCPTM,
            temperature=0.0,  # Temperatura zero para máxima fidelidade de OCR
        )

        modelos = _obter_modelos_cascata(client, modelo)

        erros_por_modelo = []
        for nome_modelo in modelos:
            try:
                response = client.models.generate_content(
                    model=nome_modelo,
                    contents=[
                        part,
                        "Extraia todos os dados legíveis da placa principal deste dormente.",
                    ],
                    config=config,
                )
                if response.parsed:
                    resposta = RespostaOcrPlaca(
                        sucesso=True,
                        timestamp_iso=datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                        resolucao_enviada_pixels=resolucao_str,
                        modelo_utilizado=nome_modelo,
                        tempo_processamento_segundos=round(time.time() - inicio, 2),
                        dados=response.parsed,
                        erro=None,
                    )
                    return (
                        resposta.model_dump(by_alias=True)
                        if retornar_dict
                        else resposta
                    )
            except (APIError, Exception) as err:
                erros_por_modelo.append(f"{nome_modelo}: {err}")

        raise RuntimeError(
            f"Todos os modelos falharam. Detalhes: {' | '.join(erros_por_modelo[:3])}"
        )

    except Exception as erro_fatal:
        resposta_erro = RespostaOcrPlaca(
            sucesso=False,
            timestamp_iso=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            resolucao_enviada_pixels=resolucao_str,
            modelo_utilizado=None,
            tempo_processamento_segundos=round(time.time() - inicio, 2),
            dados=None,
            erro=str(erro_fatal),
        )
        return (
            resposta_erro.model_dump(by_alias=True)
            if retornar_dict
            else resposta_erro
        )