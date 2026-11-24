import io
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator

load_dotenv(override=False)

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


def _apenas_digitos(valor: Optional[str]) -> Optional[str]:
    if not valor:
        return None
    limpo = re.sub(r"\D", "", valor)
    return limpo if limpo else None


class BaseCamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
    )


class ItemDof(BaseCamelModel):
    produto: Optional[str] = Field(
        None, description="Tipo de produto florestal. Ex: Madeira Serrada, Dormente."
    )
    especie_nome_popular: Optional[str] = Field(
        None, description="Nome popular da espécie. Ex: Garapeira, Maçaranduba."
    )
    especie_nome_cientifico: Optional[str] = Field(
        None, description="Nome científico da espécie. Ex: Dialium guianense."
    )
    quantidade: Optional[float] = Field(
        None, ge=0, description="Quantidade numérica do item (ex: volume em m³ ou peças)."
    )
    unidade: Optional[str] = Field(
        None, description="Unidade de medida exatamente como no documento. Ex: m3, st, un."
    )
    valor_total_reais: Optional[float] = Field(None, ge=0)


class DadosDof(BaseCamelModel):
    documento_dof_detectado: bool = Field(
        description="True se a imagem for um Documento de Origem Florestal (DOF/DOF+) do IBAMA."
    )
    confianca_geral: float = Field(
        ge=0.0,
        le=1.0,
        description="Confiança média da leitura dos campos (0.0 a 1.0).",
    )
    numero_dof: Optional[str] = Field(None, description="Número principal do DOF.")
    codigo_controle: Optional[str] = Field(
        None, description="Código de controle / chave de autenticação do DOF."
    )
    data_emissao: Optional[str] = Field(
        None, description="Data de emissão no formato YYYY-MM-DD."
    )
    data_validade: Optional[str] = Field(
        None, description="Data final de validade do transporte no formato YYYY-MM-DD."
    )
    nota_fiscal: Optional[str] = Field(None, description="Número da Nota Fiscal vinculada.")
    placa_veiculo: Optional[str] = Field(
        None, description="Placa do caminhão/veículo transportador (sem traços)."
    )
    emissor_nome: Optional[str] = Field(None, description="Razão social ou nome do remetente/origem.")
    emissor_cnpj_cpf: Optional[str] = Field(None, description="CNPJ ou CPF do emissor (apenas números).")
    emissor_ctf: Optional[str] = Field(None, description="Número do CTF/APP do emissor.")
    destinatario_nome: Optional[str] = Field(None, description="Razão social ou nome do destinatário.")
    destinatario_cnpj_cpf: Optional[str] = Field(
        None, description="CNPJ ou CPF do destinatário (apenas números)."
    )
    destinatario_ctf: Optional[str] = Field(None, description="Número do CTF/APP do destinatário.")
    itens: list[ItemDof] = Field(default_factory=list)
    volume_total_m3: Optional[float] = Field(
        None, ge=0, description="Volume total declarado em metros cúbicos (m³)."
    )
    observacao: str = Field(
        description="Aviso curto sobre campos borrados, cortados, amassados ou ilegíveis."
    )

    @model_validator(mode="after")
    def sanitizar_e_validar(self) -> "DadosDof":
        if not self.documento_dof_detectado:
            self.confianca_geral = 0.0
            self.numero_dof = None
            self.codigo_controle = None
            self.itens = []
            return self

        # Higieniza documentos numéricos para facilitar batimento no banco do backend
        self.emissor_cnpj_cpf = _apenas_digitos(self.emissor_cnpj_cpf)
        self.destinatario_cnpj_cpf = _apenas_digitos(self.destinatario_cnpj_cpf)
        if self.placa_veiculo:
            self.placa_veiculo = re.sub(r"[^A-Za-z0-9]", "", self.placa_veiculo).upper()

        return self


class RespostaOcrDof(BaseCamelModel):
    sucesso: bool
    timestamp_iso: str
    resolucao_enviada_pixels: str
    modelo_utilizado: Optional[str] = None
    tempo_processamento_segundos: float
    dados: Optional[DadosDof] = None
    erro: Optional[str] = None


SYSTEM_PROMPT_DOF = """
Você é um sistema de OCR especializado em Documento de Origem Florestal (DOF / DOF+) do IBAMA (Brasil).
Extraia os dados estruturados da foto do documento com rigor absoluto.

Regras obrigatórias:
1. Validação do Documento: Se a imagem não for um DOF (ou estiver ilegível a ponto de não confirmar), defina `documento_dof_detectado=false`.
2. Zero Alucinação: Se qualquer número (DOF, Código de Controle, CNPJ, Volume, Placa) estiver borrado, cortado, dobrado ou ilegível, retorne `null` nesse campo. NUNCA adivinhe dígitos.
3. Datas: Converta datas brasileiras (DD/MM/AAAA) para o padrão ISO `YYYY-MM-DD`.
4. Números decimais: Converta vírgulas decimais brasileiras (ex: `12,4500 m³`) para ponto decimal (`12.45`).
5. Tabela de Produtos: Extraia cada linha da tabela de produtos/espécies dentro da lista `itens`.
"""


def _preparar_imagem_baixo_consumo(
    entrada: Union[bytes, str, Path],
    max_lado: int = 1600,
) -> tuple[types.Part, tuple[int, int]]:
    """
    Otimiza a imagem para OCR de documento consumindo o mínimo de RAM:
    - Usa draft('L') para decodificar JPEG já em escala de cinza e reduzido.
    - Converte para 'L' (1 byte/pixel em vez de 3 bytes/pixel do RGB).
    - Fecha buffers explicitamente para liberar RAM imediatamente.
    """
    is_bytes = isinstance(entrada, (bytes, bytearray))
    fonte = io.BytesIO(entrada) if is_bytes else entrada

    try:
        with Image.open(fonte) as img:
            # Reduz pico de RAM na decodificação de JPEGs grandes (fotos de celular)
            try:
                img.draft("L", (max_lado, max_lado))
            except Exception:
                pass

            img = ImageOps.exif_transpose(img)
            if img.mode != "L":
                img = img.convert("L")

            img.thumbnail((max_lado, max_lado), Image.Resampling.BILINEAR)
            dimensoes = img.size

            with io.BytesIO() as buffer:
                img.save(buffer, format="JPEG", quality=85)
                bytes_otimizados = buffer.getvalue()
    finally:
        if is_bytes and hasattr(fonte, "close"):
            fonte.close()

    part = types.Part.from_bytes(data=bytes_otimizados, mime_type="image/jpeg")
    return part, dimensoes


def _obter_modelos_cascata(
    client: genai.Client, modelo_forcado: Optional[str]
) -> list[str]:
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


def ler_dof_imagem(
    imagem: Union[bytes, str, Path],
    api_key: Optional[str] = None,
    modelo: Optional[str] = None,
    timeout_ms: int = 25000,
    retornar_dict: bool = True,
) -> Union[dict, RespostaOcrDof]:
    """
    Extrai os campos estruturados de uma foto de DOF (IBAMA) com baixo consumo de RAM.
    """
    inicio = time.time()
    resolucao_str = ""

    try:
        chave = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not chave:
            raise PermissionError("Variável de ambiente GEMINI_API_KEY não configurada.")

        part, (largura, altura) = _preparar_imagem_baixo_consumo(imagem)
        resolucao_str = f"{largura}x{altura}"

        client = genai.Client(
            api_key=chave,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_DOF,
            response_mime_type="application/json",
            response_schema=DadosDof,
            temperature=0.0,
        )

        modelos = _obter_modelos_cascata(client, modelo)

        erros_por_modelo = []
        for nome_modelo in modelos:
            try:
                response = client.models.generate_content(
                    model=nome_modelo,
                    contents=[part, "Extraia todos os dados legíveis deste DOF."],
                    config=config,
                )
                if response.parsed:
                    resposta = RespostaOcrDof(
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
        resposta_erro = RespostaOcrDof(
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