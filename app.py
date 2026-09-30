"""
Vídeos de IA a partir de links da Shopee (Streamlit, pensado para usar no celular)

Fluxo por link:
  0) Lê o link da Shopee -> nome e foto do produto (plano B: você envia a foto)
  a) LLM (Gemini ou OpenAI) -> prompt_video (EN) + roteiro_voz (PT-BR)
  b) Kling AI               -> vídeo do produto (com ou sem pessoa)
  c) ElevenLabs             -> locução .mp3 + tempo de cada palavra
  d) moviepy                -> une tudo, legendas no centro, 9:16 (1080x1920)

Chaves: coloque em "Secrets" (Streamlit Cloud) ou variáveis de ambiente.
"""
from __future__ import annotations

import base64
import hmac
import io
import json
import os
import re
import shutil
import tempfile
import time
import unicodedata
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from html import unescape
from pathlib import Path
from urllib.parse import unquote

import jwt  # PyJWT (autenticação da Kling)
import numpy as np
import requests
import streamlit as st
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
    vfx,
)
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

# --------------------------------------------------------------------------
# Constantes (ajuste aqui se a API de algum provedor mudar)
# --------------------------------------------------------------------------
W, H = 1080, 1920  # saída 9:16
KLING_VIDEO_MODEL = "kling-v1-6"
KLING_TRYON_MODEL = "kolors-virtual-try-on-v1-5"
KLING_MULTI_PATH = "/v1/videos/multi-image2video"
KLING_I2V_PATH = "/v1/videos/image2video"
KLING_TRYON_PATH = "/v1/images/kolors-virtual-try-on"
ELEVEN_MODEL = "eleven_multilingual_v2"
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # troque pela voz PT-BR que preferir (segredo ELEVENLABS_VOICE_ID)
USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Mobile Safari/537.36"
)

FORMATOS = {
    "UGC": (
        "User-generated content shot on a smartphone: handheld vertical footage, natural window light, "
        "casual home setting, informal spoken tone, a practical hands-on demonstration of the product, "
        "slight imperfections that feel authentic."
    ),
    "POV": (
        "First-person point of view: the camera is the viewer's eyes, hands in frame unboxing and handling "
        "the product, macro close-ups of textures, packaging, buttons and details. "
        "Script speaks directly to 'você'."
    ),
    "REVIEW REAL": (
        "Honest review: the product is shown up close and used, explained step by step, mentioning the pros "
        "AND one fair small con, ending with a recommendation. Conversational, credible tone."
    ),
    "PROVADOR": (
        "Fashion/accessory showcase: slow 360-degree turn or camera orbit, close-ups of fabric, stitching, "
        "texture and fit; clean, well-lit setting. Script stresses fit, material and versatility."
    ),
}

SYSTEM_PROMPT = """Você é diretor criativo de vídeos curtos de e-commerce (TikTok, Reels, Shopee Vídeo).
Responda SOMENTE com um JSON válido, com exatamente estas chaves:
- "nome_produto": nome curto do produto em português (use a imagem e o nome informado).
- "prompt_video": prompt detalhado EM INGLÊS para uma IA de vídeo image-to-video. Descreva a ação, os movimentos \
de câmera, a iluminação, o cenário e o ritmo. {pessoa_regra} \
Sem texto na tela, sem logos inventados, sem alterar o formato e as cores do produto.
- "roteiro_voz": locução persuasiva em português do Brasil, natural e falada, entre 45 e 60 palavras \
(15 a 25 segundos), com gancho no início, benefícios e chamada para ação no final. \
Sem emojis e sem hashtags. Não invente preço, medidas, certificações nem características que não estejam visíveis.
Formato do vídeo: {formato}
Diretrizes do formato: {guia}"""

REGRA_COM_MODELO = (
    "A pessoa do vídeo é a modelo da foto de referência: descreva o que ela faz com o produto."
)
REGRA_SEM_MODELO = (
    "Não há modelo: NÃO mostre rosto. Mostre o produto e, se fizer sentido, apenas mãos entrando em quadro; "
    "use movimentos de câmera (órbita, aproximação, closes)."
)


@dataclass
class Config:
    formato: str
    llm_provider: str
    llm_key: str
    llm_model: str
    eleven_key: str
    voice_id: str
    kling_api_key: str
    kling_access: str
    kling_secret: str
    kling_model: str
    kling_api_key: str
    kling_base: str
    kling_mode: str
    kling_duration: str
    usar_tryon: bool


# --------------------------------------------------------------------------
# Utilitários
# --------------------------------------------------------------------------
def segredo(nome: str, padrao: str = "") -> str:
    """Lê uma chave de variável de ambiente ou dos Secrets do Streamlit."""
    valor = os.environ.get(nome)
    if valor:
        return valor
    try:
        return str(st.secrets[nome])
    except Exception:
        return padrao


def http(method: str, url: str, retries: int = 3, **kwargs) -> requests.Response:
    """requests com timeout, retentativas em 429/5xx e mensagem de erro legível."""
    kwargs.setdefault("timeout", 180)
    for tentativa in range(retries):
        try:
            resp = requests.request(method, url, **kwargs)
        except requests.RequestException as exc:
            if tentativa == retries - 1:
                raise RuntimeError(f"Falha de rede em {url.split('?')[0]}: {exc}") from exc
            time.sleep(2 * (tentativa + 1))
            continue
        if resp.status_code in (429, 500, 502, 503, 504) and tentativa < retries - 1:
            time.sleep(3 * (tentativa + 1))
            continue
        if not resp.ok:
            raise RuntimeError(f"HTTP {resp.status_code} em {url.split('?')[0]}: {resp.text[:300]}")
        return resp
    raise RuntimeError("Falha inesperada na requisição")


def preparar_imagem(raw: bytes, max_lado: int = 1280) -> bytes:
    """Converte para JPEG RGB (PNG transparente vira fundo branco) e limita o tamanho."""
    img = Image.open(io.BytesIO(raw))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        fundo = Image.new("RGB", img.size, (255, 255, 255))
        fundo.paste(img, mask=img.split()[-1])
        img = fundo
    else:
        img = img.convert("RGB")
    img.thumbnail((max_lado, max_lado))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def imagem_para_9x16(jpg: bytes, largura: int = 720, altura: int = 1280) -> bytes:
    """Coloca o produto inteiro numa tela vertical, com fundo desfocado (evita cortar o produto)."""
    img = Image.open(io.BytesIO(jpg)).convert("RGB")
    tela = ImageOps.fit(img, (largura, altura)).filter(ImageFilter.GaussianBlur(30))
    frente = ImageOps.contain(img, (largura, altura))
    tela.paste(frente, ((largura - frente.width) // 2, (altura - frente.height) // 2))
    buf = io.BytesIO()
    tela.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def slugify(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", texto.lower()).strip("-")[:50] or "produto"


# --------------------------------------------------------------------------
# 0) Ler o link da Shopee (nome + foto). Pode ser bloqueado pela Shopee.
# --------------------------------------------------------------------------
def _meta(html: str, propriedade: str) -> str:
    padroes = (
        r'<meta[^>]+(?:property|name)=["\']%s["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']%s["\']',
    )
    for padrao in padroes:
        m = re.search(padrao % re.escape(propriedade), html, re.I)
        if m:
            return unescape(m.group(1)).strip()
    return ""


def _baixar_imagem(url: str) -> bytes | None:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=25)
        if r.ok and r.content:
            Image.open(io.BytesIO(r.content)).verify()
            return r.content
    except Exception:
        pass
    return None


def ler_shopee(url: str) -> dict:
    """Devolve {'nome': str, 'imagem': bytes | None}. Nunca levanta erro por falta de foto."""
    html, candidatos = "", [url]
    try:
        resp = requests.get(
            url, headers={"User-Agent": USER_AGENT, "Accept-Language": "pt-BR,pt;q=0.9"},
            timeout=25, allow_redirects=True,
        )
        candidatos = [unquote(r.url) for r in resp.history] + [unquote(resp.url), url]
        if resp.ok:
            html = resp.text
    except requests.RequestException:
        pass

    shopid = itemid = ""
    nome = ""
    for alvo in candidatos + [html]:
        m = re.search(r"-i\.(\d+)\.(\d+)", alvo) or re.search(r"/product/(\d+)/(\d+)", alvo)
        if m:
            shopid, itemid = m.group(1), m.group(2)
            break
    for alvo in candidatos:
        m = re.search(r"shopee\.com\.br/([^/?#]+?)-i\.\d+\.\d+", alvo)
        if m:
            nome = m.group(1).replace("-", " ").strip()
            break

    titulo = re.sub(r"\s*\|\s*Shopee.*$", "", _meta(html, "og:title"))
    nome = titulo or nome

    # 1º: imagem das tags da página
    imagem = _baixar_imagem(_meta(html, "og:image")) if _meta(html, "og:image") else None

    # 2º: API pública da Shopee (costuma exigir navegador; pode falhar)
    if imagem is None and shopid and itemid:
        try:
            r = requests.get(
                "https://shopee.com.br/api/v4/item/get",
                params={"itemid": itemid, "shopid": shopid},
                headers={"User-Agent": USER_AGENT, "Referer": candidatos[0]}, timeout=20,
            )
            dados = (r.json().get("data") or {}) if r.ok else {}
            dados = dados.get("item") or dados
            nome = nome or dados.get("name") or ""
            img_id = dados.get("image") or next(iter(dados.get("images") or []), None)
            if img_id:
                for base in ("https://down-br.img.susercontent.com/file/", "https://cf.shopee.com.br/file/"):
                    imagem = _baixar_imagem(base + img_id)
                    if imagem:
                        break
        except Exception:
            pass

    return {"nome": nome, "imagem": imagem}


# --------------------------------------------------------------------------
# a) Roteiro e prompts (LLM)
# --------------------------------------------------------------------------
def gerar_roteiro(cfg: Config, nome: str, produto_jpg: bytes, com_modelo: bool) -> dict:
    system = SYSTEM_PROMPT.format(
        formato=cfg.formato, guia=FORMATOS[cfg.formato],
        pessoa_regra=REGRA_COM_MODELO if com_modelo else REGRA_SEM_MODELO,
    )
    user = f"Nome do produto: {nome}. Gere o JSON."

    if cfg.llm_provider == "OpenAI":
        client = OpenAI(api_key=cfg.llm_key, timeout=120)
        resp = client.chat.completions.create(
            model=cfg.llm_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + b64(produto_jpg)},
                        },
                    ],
                },
            ],
        )
        texto = resp.choices[0].message.content
    else:  # Gemini via REST
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.llm_model}:generateContent"
        corpo = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user},
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64(produto_jpg)}},
                    ],
                }
            ],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        resp = http(
            "POST", url, json=corpo,
            headers={"x-goog-api-key": cfg.llm_key, "Content-Type": "application/json"},
        ).json()
        texto = resp["candidates"][0]["content"]["parts"][0]["text"]

    texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto.strip())
    dados = json.loads(texto)
    for chave in ("prompt_video", "roteiro_voz"):
        if not dados.get(chave):
            raise RuntimeError(f"O LLM não retornou a chave '{chave}'")
    dados.setdefault("nome_produto", nome)
    return dados


# --------------------------------------------------------------------------
# b) Vídeo (Kling AI)
# --------------------------------------------------------------------------
def kling_headers(cfg: Config) -> dict:
    if cfg.kling_api_key:  # padrão novo: uma única API Key
        return {"Authorization": f"Bearer {cfg.kling_api_key}", "Content-Type": "application/json"}
    agora = int(time.time())  # padrão antigo: Access Key + Secret Key
    token = jwt.encode(
        {"iss": cfg.kling_access, "exp": agora + 1800, "nbf": agora - 5},
        cfg.kling_secret,
        algorithm="HS256",
        headers={"alg": "HS256", "typ": "JWT"},
    )
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def kling_enviar(cfg: Config, path: str, corpo: dict) -> str:
    resp = http("POST", cfg.kling_base + path, headers=kling_headers(cfg), json=corpo).json()
    if resp.get("code") != 0:
        raise RuntimeError(f"Kling recusou o pedido: {resp.get('message')}")
    return resp["data"]["task_id"]


def kling_aguardar(cfg: Config, path: str, task_id: str, timeout: int = 900) -> dict:
    inicio = time.time()
    while time.time() - inicio < timeout:
        resp = http("GET", f"{cfg.kling_base}{path}/{task_id}", headers=kling_headers(cfg)).json()
        dados = resp.get("data") or {}
        situacao = dados.get("task_status")
        if situacao == "succeed":
            return dados["task_result"]
        if situacao == "failed":
            raise RuntimeError(f"Kling falhou: {dados.get('task_status_msg')}")
        time.sleep(8)
    raise RuntimeError("Kling: tempo esgotado aguardando o vídeo")


def gerar_video(cfg: Config, produto: bytes, modelo: bytes | None, prompt: str, log) -> str:
    """Retorna a URL do vídeo gerado."""
    negativo = "blurry, distorted, extra fingers, deformed product, text, watermark, logo"

    # Sem pessoa: anima só o produto (imagem colocada em tela vertical)
    if modelo is None:
        log("2/4 Animando o produto…")
        tid = kling_enviar(cfg, KLING_I2V_PATH, {
            "model_name": cfg.kling_model,
            "mode": cfg.kling_mode,
            "duration": cfg.kling_duration,
            "image": b64(imagem_para_9x16(produto)),
            "prompt": prompt[:2400],
            "negative_prompt": negativo,
        })
        return kling_aguardar(cfg, KLING_I2V_PATH, tid)["videos"][0]["url"]

    # Com pessoa e PROVADOR: try-on opcional
    if cfg.formato == "PROVADOR" and cfg.usar_tryon:
        try:
            log("2/4 Provador virtual (try-on)…")
            tid = kling_enviar(cfg, KLING_TRYON_PATH, {
                "model_name": KLING_TRYON_MODEL,
                "human_image": b64(modelo),
                "cloth_image": b64(produto),
            })
            res = kling_aguardar(cfg, KLING_TRYON_PATH, tid, timeout=300)
            imagem = preparar_imagem(http("GET", res["images"][0]["url"]).content)
            log("2/4 Animando a imagem do try-on…")
            tid = kling_enviar(cfg, KLING_I2V_PATH, {
                "model_name": cfg.kling_model,
                "mode": cfg.kling_mode,
                "duration": cfg.kling_duration,
                "image": b64(imagem),
                "prompt": prompt[:2400],
            })
            return kling_aguardar(cfg, KLING_I2V_PATH, tid)["videos"][0]["url"]
        except Exception as exc:  # cai para o modo multi-imagem
            log(f"2/4 Try-on falhou ({str(exc)[:80]}); usando modo multi-imagem…")

    log("2/4 Gerando vídeo (pessoa + produto)…")
    prompt_final = (
        f"{prompt} Image 1 is the person, image 2 is the product. Keep the person's face and the "
        "product's shape, colors and details identical to the reference images."
    )
    tid = kling_enviar(cfg, KLING_MULTI_PATH, {
        "model_name": cfg.kling_model,
        "image_list": [{"image": b64(modelo)}, {"image": b64(produto)}],
        "prompt": prompt_final[:2400],
        "negative_prompt": negativo,
        "mode": cfg.kling_mode,
        "duration": cfg.kling_duration,
        "aspect_ratio": "9:16",
    })
    return kling_aguardar(cfg, KLING_MULTI_PATH, tid)["videos"][0]["url"]


# --------------------------------------------------------------------------
# c) Locução (ElevenLabs) com timestamps por palavra
# --------------------------------------------------------------------------
def palavras_do_alinhamento(al: dict) -> list[tuple[str, float, float]]:
    palavras, atual, t0, t1 = [], "", 0.0, 0.0
    for c, ini, fim in zip(
        al["characters"], al["character_start_times_seconds"], al["character_end_times_seconds"]
    ):
        if c.isspace():
            if atual:
                palavras.append((atual, t0, t1))
                atual = ""
        else:
            if not atual:
                t0 = ini
            atual += c
            t1 = fim
    if atual:
        palavras.append((atual, t0, t1))
    return palavras


def gerar_audio(cfg: Config, texto: str, pasta: Path):
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{cfg.voice_id}/with-timestamps"
        "?output_format=mp3_44100_128"
    )
    resp = http(
        "POST", url,
        headers={"xi-api-key": cfg.eleven_key, "Content-Type": "application/json"},
        json={
            "text": texto,
            "model_id": ELEVEN_MODEL,
            "voice_settings": {
                "stability": 0.45, "similarity_boost": 0.8, "style": 0.3, "use_speaker_boost": True,
            },
        },
    ).json()
    mp3 = pasta / "locucao.mp3"
    mp3.write_bytes(base64.b64decode(resp["audio_base64"]))
    return mp3, palavras_do_alinhamento(resp["alignment"])


# --------------------------------------------------------------------------
# d) Edição e renderização final (moviepy, local)
# --------------------------------------------------------------------------
@lru_cache(maxsize=32)
def achar_fonte(tamanho: int):
    candidatas = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for caminho in candidatas:
        if os.path.exists(caminho):
            return ImageFont.truetype(caminho, tamanho)
    try:
        return ImageFont.load_default(size=tamanho)
    except TypeError:
        return ImageFont.load_default()


def desenhar_legenda(palavras: list[str], ativo: int) -> np.ndarray:
    """Desenha o grupo de palavras; a palavra ativa fica em amarelo."""
    palavras = [p.upper() for p in palavras]
    medidor = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    tamanho = 96
    while True:
        fonte = achar_fonte(tamanho)
        espaco = medidor.textlength(" ", font=fonte)
        larguras = [medidor.textlength(p, font=fonte) for p in palavras]
        total = sum(larguras) + espaco * (len(palavras) - 1)
        if total <= W - 120 or tamanho <= 40:
            break
        tamanho -= 6
    img = Image.new("RGBA", (W, int(tamanho * 2)), (0, 0, 0, 0))
    desenho = ImageDraw.Draw(img)
    x, y = (W - total) / 2, tamanho * 0.4
    for i, (palavra, largura) in enumerate(zip(palavras, larguras)):
        cor = (255, 214, 0, 255) if i == ativo else (255, 255, 255, 255)
        desenho.text(
            (x, y), palavra, font=fonte, fill=cor,
            stroke_width=max(4, tamanho // 10), stroke_fill=(0, 0, 0, 255),
        )
        x += largura + espaco
    return np.array(img)


def clips_de_legenda(palavras: list[tuple[str, float, float]]) -> list:
    grupos, atual = [], []
    for p in palavras:
        if atual and (len(atual) >= 3 or sum(len(x[0]) for x in atual) + len(p[0]) > 16):
            grupos.append(atual)
            atual = []
        atual.append(p)
    if atual:
        grupos.append(atual)

    clips = []
    for gi, grupo in enumerate(grupos):
        textos = [w[0] for w in grupo]
        inicio_proximo = grupos[gi + 1][0][1] if gi + 1 < len(grupos) else None
        for i, (_, ini, fim_palavra) in enumerate(grupo):
            if i + 1 < len(grupo):
                fim = grupo[i + 1][1]
            elif inicio_proximo is not None:
                fim = min(inicio_proximo, fim_palavra + 0.35)
            else:
                fim = fim_palavra + 0.35
            if fim - ini < 0.05:
                continue
            clip = (
                ImageClip(desenhar_legenda(textos, i))
                .with_duration(fim - ini)
                # efeito "pop": a legenda entra levemente maior e assenta em 0,1 s
                .resized(lambda t: 1 + 0.10 * max(0.0, 1 - t / 0.10))
                .with_start(ini)
                .with_position(("center", "center"))
            )
            clips.append(clip)
    return clips


def estender(clip, duracao: float):
    """Cobre a duração da locução com vai-e-volta (boomerang) se o clipe for curto."""
    if clip.duration >= duracao:
        return clip.subclipped(0, duracao)
    vai_volta = concatenate_videoclips([clip, clip.with_effects([vfx.TimeMirror()])])
    return vai_volta.with_effects([vfx.Loop(duration=duracao)])


def cobrir_9x16(clip):
    """Escala e corta ao centro para exatamente 1080x1920."""
    escala = max(W / clip.w, H / clip.h)
    clip = clip.resized((max(W, round(clip.w * escala)), max(H, round(clip.h * escala))))
    x, y = int((clip.w - W) / 2), int((clip.h - H) / 2)
    return clip.cropped(x1=x, y1=y, x2=x + W, y2=y + H)


def renderizar(video: Path, mp3: Path, palavras, saida: Path) -> None:
    audio = AudioFileClip(str(mp3))
    duracao = audio.duration + 0.4
    base = VideoFileClip(str(video)).without_audio()
    base = cobrir_9x16(estender(base, duracao))
    final = (
        CompositeVideoClip([base] + clips_de_legenda(palavras), size=(W, H))
        .with_audio(audio)
        .with_duration(duracao)
    )
    final.write_videofile(
        str(saida), fps=30, codec="libx264", audio_codec="aac",
        preset="medium", threads=2, logger=None,
    )
    final.close()
    base.close()
    audio.close()


# --------------------------------------------------------------------------
# Esteira completa de um produto (roda em thread; não usa st.*)
# --------------------------------------------------------------------------
def processar_produto(idx: int, item: dict, modelo_raw: bytes | None,
                      cfg: Config, workdir: Path, status: dict) -> dict:
    def log(msg: str):
        status[idx] = msg

    nome = item["nome"] or item["link"] or "produto"
    try:
        imagem = item["imagem"]
        if imagem is None:
            log("0/4 Lendo o link da Shopee…")
            info = ler_shopee(item["link"])
            nome = item["nome"] or info["nome"] or "produto"
            imagem = info["imagem"]
            if imagem is None:
                raise RuntimeError(
                    "Não consegui pegar a foto deste link (a Shopee às vezes bloqueia). "
                    "Salve a foto do produto no celular e envie em 'Plano B'."
                )

        pasta = workdir / f"{idx:02d}"
        pasta.mkdir(parents=True, exist_ok=True)
        produto = preparar_imagem(imagem)
        modelo = preparar_imagem(modelo_raw) if modelo_raw else None

        log("1/4 Escrevendo roteiro e prompt…")
        dados = gerar_roteiro(cfg, nome, produto, com_modelo=modelo is not None)

        url_video = gerar_video(cfg, produto, modelo, dados["prompt_video"], log)
        clipe = pasta / "clipe.mp4"
        clipe.write_bytes(http("GET", url_video).content)

        log("3/4 Gerando locução…")
        mp3, palavras = gerar_audio(cfg, dados["roteiro_voz"], pasta)

        log("4/4 Montando vídeo final (legendas + 9:16)…")
        arquivo = f"{idx + 1:02d}-{slugify(dados['nome_produto'])}.mp4"
        saida = pasta / arquivo
        renderizar(clipe, mp3, palavras, saida)

        log("✅ Pronto")
        return {
            "nome": dados["nome_produto"], "arquivo": arquivo, "video": saida.read_bytes(),
            "prompt": dados["prompt_video"], "roteiro": dados["roteiro_voz"], "erro": None,
        }
    except Exception as exc:
        log(f"❌ Erro: {str(exc)[:120]}")
        return {"nome": nome, "arquivo": None, "video": None, "prompt": "", "roteiro": "", "erro": str(exc)}


# --------------------------------------------------------------------------
# Interface Streamlit (uma coluna, boa para celular)
# --------------------------------------------------------------------------
def montar_zip(resultados: list[dict]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        roteiros = []
        for r in resultados:
            z.writestr(r["arquivo"], r["video"])
            roteiros.append(f"## {r['arquivo']}\n{r['roteiro']}\n")
        z.writestr("roteiros.txt", "\n".join(roteiros))
    return buf.getvalue()


def pedir_senha() -> None:
    senha = segredo("APP_PASSWORD")
    if not senha or st.session_state.get("liberado"):
        return
    digitada = st.text_input("🔒 Senha do app", type="password")
    if digitada:
        if hmac.compare_digest(digitada, senha):
            st.session_state["liberado"] = True
            st.rerun()
        st.error("Senha incorreta.")
    st.stop()


def coletar_chaves() -> dict:
    """Usa as chaves dos Secrets; só mostra campos para o que estiver faltando."""
    nomes = ["GEMINI_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY",
             "KLING_API_KEY", "KLING_ACCESS_KEY", "KLING_SECRET_KEY"]
    chaves = {n: segredo(n) for n in nomes}
    sem_llm = not (chaves["GEMINI_API_KEY"] or chaves["OPENAI_API_KEY"])
    outros = {"ELEVENLABS_API_KEY": "Chave ElevenLabs"}
    faltam = [n for n in outros if not chaves[n]]
    sem_kling = not (chaves["KLING_API_KEY"] or (chaves["KLING_ACCESS_KEY"] and chaves["KLING_SECRET_KEY"]))
    if sem_llm or faltam or sem_kling:
        with st.expander("🔑 Chaves de API (preencha uma vez)", expanded=True):
            if sem_llm:
                chaves["GEMINI_API_KEY"] = st.text_input("Chave Gemini", type="password")
                chaves["OPENAI_API_KEY"] = st.text_input("ou chave OpenAI", type="password")
            for n in faltam:
                chaves[n] = st.text_input(outros[n], type="password")
            if sem_kling:
                chaves["KLING_API_KEY"] = st.text_input("Chave Kling (API Key)", type="password")
    return chaves


def main():
    st.set_page_config(page_title="Vídeos da Shopee com IA", page_icon="🎬", layout="centered")
    st.title("🎬 Vídeos da Shopee com IA")
    pedir_senha()

    chaves = coletar_chaves()

    st.subheader("1. Cole o link do produto")
    texto_links = st.text_area(
        "Links da Shopee (um por linha; pode colar a mensagem de compartilhar inteira)",
        height=120, placeholder="https://s.shopee.com.br/...",
    )
    with st.expander("Plano B: enviar a foto do produto (se o link não funcionar)"):
        fotos = st.file_uploader("Fotos dos produtos", type=["png", "jpg", "jpeg", "webp"],
                                 accept_multiple_files=True)
    with st.expander("Quer uma pessoa no vídeo? (opcional)"):
        modelo = st.file_uploader("Foto da pessoa", type=["png", "jpg", "jpeg", "webp"])
        autorizado = st.checkbox(
            "Tenho autorização da pessoa da foto e vou identificar o vídeo como gerado por IA "
            "quando a plataforma exigir."
        )

    st.subheader("2. Escolha o estilo")
    formato = st.selectbox("Formato do vídeo", list(FORMATOS.keys()))

    with st.expander("Opções avançadas"):
        duracao = st.selectbox("Duração do clipe da IA (s)", ["10", "5"])
        qualidade = st.selectbox("Qualidade", ["std", "pro"])
        paralelo = st.slider("Vídeos ao mesmo tempo", 1, 3, 1)
        usar_tryon = st.checkbox("Provador virtual (só com pessoa + roupa)") if formato == "PROVADOR" else False

    st.subheader("3. Gere")
    if st.button("🚀 Gerar vídeos", type="primary", use_container_width=True):
        links = [u.rstrip(".,;)") for u in re.findall(r"https?://\S+", texto_links)]
        itens = [{"nome": "", "link": u, "imagem": None} for u in links]
        for f in fotos or []:
            itens.append({"nome": Path(f.name).stem.replace("_", " ").replace("-", " "),
                          "link": None, "imagem": f.getvalue()})

        faltando = []
        if not itens:
            faltando.append("um link da Shopee (ou uma foto)")
        if not (chaves["GEMINI_API_KEY"] or chaves["OPENAI_API_KEY"]):
            faltando.append("chave Gemini ou OpenAI")
        if not chaves["ELEVENLABS_API_KEY"]:
            faltando.append("chave ElevenLabs")
        if not (chaves["KLING_API_KEY"] or (chaves["KLING_ACCESS_KEY"] and chaves["KLING_SECRET_KEY"])):
            faltando.append("chave da Kling")
        if modelo and not autorizado:
            faltando.append("confirmação de autorização da pessoa da foto")
        if faltando:
            st.error("Falta: " + ", ".join(faltando))
            return

        usar_gemini = bool(chaves["GEMINI_API_KEY"])
        cfg = Config(
            formato=formato,
            llm_provider="Gemini" if usar_gemini else "OpenAI",
            llm_key=chaves["GEMINI_API_KEY"] if usar_gemini else chaves["OPENAI_API_KEY"],
            llm_model=segredo("LLM_MODEL", "gemini-2.5-flash" if usar_gemini else "gpt-4o-mini"),
            eleven_key=chaves["ELEVENLABS_API_KEY"],
            voice_id=segredo("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID).strip(),
            kling_api_key=chaves["KLING_API_KEY"],
            kling_access=chaves["KLING_ACCESS_KEY"],
            kling_secret=chaves["KLING_SECRET_KEY"],
            kling_model=segredo("KLING_MODEL", KLING_VIDEO_MODEL),
            kling_base=segredo("KLING_BASE_URL", "https://api-singapore.klingai.com").rstrip("/"),
            kling_mode=qualidade,
            kling_duration=duracao,
            usar_tryon=usar_tryon,
        )

        modelo_raw = modelo.getvalue() if modelo else None
        n = len(itens)
        rotulos = [it["nome"] or f"Link {i + 1}" for i, it in enumerate(itens)]
        workdir = Path(tempfile.mkdtemp(prefix="videos_"))
        status = {i: "na fila" for i in range(n)}
        barra = st.progress(0.0, text=f"0/{n} concluídos")
        painel = st.empty()

        with ThreadPoolExecutor(max_workers=paralelo) as pool:
            futuros = [
                pool.submit(processar_produto, i, item, modelo_raw, cfg, workdir, status)
                for i, item in enumerate(itens)
            ]
            while True:
                feitos = sum(f.done() for f in futuros)
                barra.progress(feitos / n, text=f"{feitos}/{n} concluídos")
                painel.markdown("\n".join(f"- **{rotulos[i]}** — {status[i]}" for i in range(n)))
                if feitos == n:
                    break
                time.sleep(1.5)
            st.session_state["resultados"] = [f.result() for f in futuros]

        shutil.rmtree(workdir, ignore_errors=True)

    resultados = st.session_state.get("resultados")
    if resultados:
        ok = [r for r in resultados if not r["erro"]]
        for r in resultados:
            if r["erro"]:
                st.error(f"{r['nome']}: {r['erro']}")
        for i, r in enumerate(ok):
            st.divider()
            st.video(r["video"])
            st.caption(r["nome"])
            st.download_button("⬇️ Baixar vídeo (MP4)", r["video"], file_name=r["arquivo"],
                               mime="video/mp4", key=f"dl_{i}", use_container_width=True)
            with st.expander("Roteiro e prompt"):
                st.write(r["roteiro"])
                st.code(r["prompt"], language=None)
        if len(ok) > 1:
            st.download_button("⬇️ Baixar todos (.ZIP)", montar_zip(ok),
                               file_name="videos_shopee.zip", mime="application/zip",
                               type="primary", use_container_width=True)


if __name__ == "__main__":
    main()
