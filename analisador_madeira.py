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

# Carrega .env apenas se existir (desenvolvimento local).
# Em produção (Azure App Service), as variáveis vêm do ambiente do sistema.
load_dotenv(override=False)

MAX_IMAGENS_POR_INSPECAO = 5

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


# --- 1. Contratos de Dados (camelCase para o contrato do jade-backend) ---
def _to_camel(string: str) -> str:
    partes = string.split("_")
    return partes[0] + "".join(p.capitalize() for p in partes[1:])


class BaseCamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
    )


class DeteccaoDefeito(BaseCamelModel):
    """Representa um achado individual compatível com app/business_rules.py."""

    tipo_defeito: Literal[
        "fenda_rachadura",
        "no_firme",
        "no_solto_cariado",
        "empenamento",
        "apodrecimento",
        "outro",
    ] = Field(description="Categoria do defeito físico encontrado no dormente.")
    severidade: Literal["leve", "moderada", "critica"] = Field(
        description="Severidade para aplicação das regras de veredito."
    )
    confianca: float = Field(
        ge=0.0,
        le=1.0,
        description="Confiança estimada da detecção (0.0 a 1.0).",
    )
    localizacao: Literal[
        "topo_extremidade",
        "centro_face",
        "bordas",
        "multiplas_regioes",
        "indeterminado",
    ]
    comprimento_pixels: Optional[float] = Field(None, ge=0)
    largura_pixels: Optional[float] = Field(None, ge=0)
    comprimento_cm: Optional[float] = Field(
        None,
        ge=0,
        description="Estimativa em cm (calculada automaticamente se houver régua de 30 cm).",
    )
    largura_cm: Optional[float] = Field(
        None,
        ge=0,
        description="Estimativa em cm (calculada automaticamente se houver régua de 30 cm).",
    )
    observacao: str = Field(description="Descrição técnica curta do defeito.")


class CalibracaoEscala(BaseCamelModel):
    regua_30cm_detectada: bool = Field(
        description="True se houver régua de 30 cm visível; False caso contrário."
    )
    comprimento_regua_pixels: Optional[float] = Field(
        None,
        ge=0,
        description="Comprimento ocupado pelos 30 cm da régua em pixels.",
    )
    fator_conversao_px_para_cm: Optional[float] = Field(
        None,
        description="Calculado automaticamente (30.0 / comprimento_regua_pixels).",
    )
    comprimento_peca_pixels: Optional[float] = Field(None, ge=0)
    largura_peca_pixels: Optional[float] = Field(None, ge=0)
    comprimento_peca_cm: Optional[float] = Field(None, ge=0)
    largura_peca_cm: Optional[float] = Field(None, ge=0)
    observacao_escala: str = Field(
        description="Nota curta sobre a referência de escala utilizada."
    )


class ResultadoAnaliseIA(BaseCamelModel):
    dormente_detectado: bool = Field(
        description="True se houver dormente/peça de madeira visível. False evita falso APROVADO (Bug #204)."
    )
    qualidade_imagem_suficiente: bool = Field(
        description="False apenas se a imagem estiver muito escura, tremida ou obstruída."
    )
    calibracao: CalibracaoEscala
    deteccoes: list[DeteccaoDefeito] = Field(default_factory=list)
    veredito_sugerido: Literal[
        "aprovado",
        "quarentena",
        "rejeitado",
        "analise_necessaria",
    ] = Field(description="Calculado automaticamente via regra de negócio.")
    parecer_tecnico: str = Field(
        description="Parecer técnico objetivo em até 3 frases, sem saudações."
    )

    @model_validator(mode="after")
    def processar_escala_e_veredito(self) -> "ResultadoAnaliseIA":
        # 1. Conversão de escala (pixels -> cm) com arredondamento realista (1 casa decimal)
        esc = self.calibracao
        if (
            esc.regua_30cm_detectada
            and esc.comprimento_regua_pixels is not None
            and esc.comprimento_regua_pixels > 0
        ):
            fator = 30.0 / esc.comprimento_regua_pixels
            esc.fator_conversao_px_para_cm = round(fator, 5)

            if esc.comprimento_peca_pixels is not None:
                esc.comprimento_peca_cm = round(esc.comprimento_peca_pixels * fator, 1)
            if esc.largura_peca_pixels is not None:
                esc.largura_peca_cm = round(esc.largura_peca_pixels * fator, 1)

            for det in self.deteccoes:
                if det.comprimento_pixels is not None:
                    det.comprimento_cm = round(det.comprimento_pixels * fator, 1)
                if det.largura_pixels is not None:
                    det.largura_cm = round(det.largura_pixels * fator, 1)
        else:
            esc.regua_30cm_detectada = False
            esc.comprimento_regua_pixels = None
            esc.fator_conversao_px_para_cm = None
            esc.comprimento_peca_cm = None
            esc.largura_peca_cm = None
            for det in self.deteccoes:
                det.comprimento_cm = None
                det.largura_cm = None

        # 2. Espelho das regras de negócio (app/business_rules.py + Bug #204)
        if not self.dormente_detectado or not self.qualidade_imagem_suficiente:
            self.veredito_sugerido = "analise_necessaria"
            return self

        criticas = [
            d for d in self.deteccoes if d.severidade == "critica" and d.confianca >= 0.40
        ]
        moderadas = [
            d for d in self.deteccoes if d.severidade == "moderada" and d.confianca >= 0.40
        ]
        leves = [
            d for d in self.deteccoes if d.severidade == "leve" and d.confianca >= 0.40
        ]

        if criticas:
            self.veredito_sugerido = "rejeitado"
        elif moderadas or len(leves) >= 3:
            self.veredito_sugerido = "quarentena"
        else:
            self.veredito_sugerido = "aprovado"

        return self


class RespostaMotorIA(BaseCamelModel):
    sucesso: bool
    timestamp_iso: str
    resolucao_enviada_pixels: list[str]
    modelo_utilizado: Optional[str] = None
    tempo_processamento_segundos: float
    analise: Optional[ResultadoAnaliseIA] = None
    erro: Optional[str] = None


# --- 2. Prompt Técnico ---
SYSTEM_PROMPT = """
Você é um inspetor técnico sênior de qualidade de dormentes de madeira serrada e beneficiada.
Analise as imagens fornecidas buscando não conformidades físicas (fendas/rachaduras, nós, apodrecimento e empenamento).

Regras obrigatórias:
1. Detecção do Dormente: Se não houver um dormente ou peça de madeira claramente visível na foto, defina `dormente_detectado=false` e retorne `deteccoes=[]`.
2. Detecções Individuais: Liste cada defeito relevante encontrado em `deteccoes`, atribuindo `severidade` ('leve', 'moderada' ou 'critica') e `confianca` realista entre 0.0 e 1.0.
3. Medição em Pixels Primeiro: Use a resolução informada (largura x altura px) como referência para estimar as dimensões em pixels.
4. Régua de 30 cm Opcional: Se houver uma régua de 30 cm visível, defina `regua_30cm_detectada=true` e informe `comprimento_regua_pixels`. O sistema converterá automaticamente para cm. Se não houver régua, siga a inspeção normalmente.
5. Diferenciação: Não confunda veios naturais escuros, manchas superficiais ou marcas normais de serra com rachaduras estruturais.
6. Empenamento: Só registre empenamento se o ângulo da foto (borda/topo ou referência plana) permitir comprovar a deformação.
"""


# --- 3. Funções Internas (100% em memória) ---
def _preparar_imagem_em_memoria(
    entrada: Union[bytes, str, Path],
    max_lado: int = 1024,
) -> tuple[types.Part, tuple[int, int]]:
    """Corrige orientação EXIF e redimensiona a imagem em RAM sem tocar no disco."""
    fonte = io.BytesIO(entrada) if isinstance(entrada, (bytes, bytearray)) else entrada

    with Image.open(fonte) as img:
        img_corrigida = ImageOps.exif_transpose(img).convert("RGB")
        img_corrigida.thumbnail((max_lado, max_lado), Image.Resampling.LANCZOS)
        dimensoes = img_corrigida.size  # (largura, altura)

        buffer = io.BytesIO()
        img_corrigida.save(buffer, format="JPEG", quality=85)
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

        fila = [m for m in MODELOS_PREFERIDOS if m in disponiveis] + [
            m for m in disponiveis if m not in MODELOS_PREFERIDOS
        ]
        _CACHE_MODELOS_DISPONIVEIS = fila if fila else MODELOS_PREFERIDOS

    return _CACHE_MODELOS_DISPONIVEIS


# --- 4. Interface Pública para o Backend ---
def analisar_madeira(
    imagens: list[Union[bytes, str, Path]],
    api_key: Optional[str] = None,
    modelo: Optional[str] = None,
    timeout_ms: int = 25000,
    retornar_dict: bool = True,
) -> Union[dict, RespostaMotorIA]:
    """
    Executa a análise visual de até 5 imagens de dormentes de madeira.
    """
    inicio = time.time()
    resolucoes: list[str] = []

    try:
        if not imagens:
            raise ValueError("Nenhuma imagem fornecida para análise.")
        if len(imagens) > MAX_IMAGENS_POR_INSPECAO:
            raise ValueError(
                f"Limite excedido: envie no máximo {MAX_IMAGENS_POR_INSPECAO} imagens por inspeção."
            )

        chave = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not chave:
            raise PermissionError("Variável de ambiente GEMINI_API_KEY não configurada.")

        parts: list[types.Part] = []
        detalhes_resolucao: list[str] = []

        for idx, item in enumerate(imagens, start=1):
            part, (largura, altura) = _preparar_imagem_em_memoria(item)
            parts.append(part)
            resolucoes.append(f"{largura}x{altura}")
            detalhes_resolucao.append(
                f"- Imagem {idx}: resolução exata de {largura}x{altura} pixels."
            )

        client = genai.Client(
            api_key=chave,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ResultadoAnaliseIA,
            temperature=0.1,
        )

        prompt_usuario = (
            "Realize a inspeção física deste dormente de madeira.\n"
            "Resoluções das imagens enviadas:\n"
            + "\n".join(detalhes_resolucao)
            + "\n\nIdentifique os defeitos individuais com confiança (0.0 a 1.0) e estime as dimensões em pixels. "
            "Se houver régua de 30 cm visível, informe o comprimento dela em pixels."
        )

        conteudo = [*parts, prompt_usuario]
        modelos = _obter_modelos_cascata(client, modelo)

        erros_por_modelo = []
        for nome_modelo in modelos:
            try:
                response = client.models.generate_content(
                    model=nome_modelo,
                    contents=conteudo,
                    config=config,
                )
                if response.parsed:
                    resposta = RespostaMotorIA(
                        sucesso=True,
                        timestamp_iso=datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                        resolucao_enviada_pixels=resolucoes,
                        modelo_utilizado=nome_modelo,
                        tempo_processamento_segundos=round(time.time() - inicio, 2),
                        analise=response.parsed,
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
        resposta_erro = RespostaMotorIA(
            sucesso=False,
            timestamp_iso=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            resolucao_enviada_pixels=resolucoes,
            modelo_utilizado=None,
            tempo_processamento_segundos=round(time.time() - inicio, 2),
            analise=None,
            erro=str(erro_fatal),
        )
        return (
            resposta_erro.model_dump(by_alias=True)
            if retornar_dict
            else resposta_erro
        )