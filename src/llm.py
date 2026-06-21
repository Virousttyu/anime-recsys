"""Groq LLM client + 動漫推薦相關的 prompt 模板。

設計重點:
- 全部用 @st.cache_data 快取結果,同樣參數的 prompt 不重複呼叫 API。
- API key 從 Streamlit secrets 或環境變數讀,沒有就回 None,不會炸 app。
- 三個 prompt 函式對應 UI 三個位置:
    * top_pick_pitch       — Feature A 三大首推的卡片副本(40 字)
    * top10_explanation    — 推薦結果分頁的整體解釋(100-150 字)
    * discovery_pitch      — Feature B 探索式推薦的單張卡片副本(60 字)
"""
from __future__ import annotations

import os
from typing import Optional

import streamlit as st

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

DEFAULT_MODEL = "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# Configuration / availability
# ---------------------------------------------------------------------------

def get_api_key() -> Optional[str]:
    """從 Streamlit secrets 或環境變數讀 API key。"""
    try:
        return st.secrets["groq"]["api_key"]
    except (KeyError, FileNotFoundError, AttributeError):
        pass
    return os.environ.get("GROQ_API_KEY", None)


def is_configured() -> bool:
    """Groq client 是否可用 (套件已裝 + API key 已設定)。"""
    return GROQ_AVAILABLE and bool(get_api_key())


@st.cache_resource
def _get_client():
    key = get_api_key()
    if not (GROQ_AVAILABLE and key):
        return None
    return Groq(api_key=key)


# ---------------------------------------------------------------------------
# Low-level LLM call (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _chat(prompt: str, model: str = DEFAULT_MODEL,
          max_tokens: int = 200, temperature: float = 0.7) -> str:
    """呼叫 Groq chat completion。失敗時回傳簡短錯誤字串而非丟例外。"""
    client = _get_client()
    if client is None:
        return ""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM 呼叫失敗:{type(e).__name__}]"


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

def _shorten(text, n):
    if not isinstance(text, str):
        return ""
    return text[:n].strip()


def top_pick_pitch(anime_title: str, anime_genre: str,
                   anime_synopsis: str, user_likes_titles: list) -> str:
    """Feature A 三大首推卡片副本 (~40 字)。"""
    if not is_configured():
        return ""
    likes = ", ".join(f"《{t}》" for t in user_likes_titles[:5]) or "未知"
    prompt = f"""你是動漫推薦師。請用一句話 (40 字內,繁體中文) 說明為什麼《{anime_title}》適合這位使用者。

使用者喜歡的作品: {likes}
推薦動漫類型: {anime_genre}
劇情概要: {_shorten(anime_synopsis, 250)}

請只回覆一句話,不要前言、不要重複問題。範例:「結合了你愛的機甲與政治深度,但加入了更鮮明的情感弧線。」"""
    return _chat(prompt, max_tokens=120, temperature=0.85)


def top10_explanation(rec_titles: list, user_likes_titles: list) -> str:
    """整體推薦清單解釋 (~100-150 字)。"""
    if not is_configured():
        return ""
    if not rec_titles:
        return ""
    titles = ", ".join(f"《{t}》" for t in rec_titles[:5])
    likes = ", ".join(f"《{t}》" for t in user_likes_titles[:5]) or "未知"
    prompt = f"""你是動漫推薦師。請用 100-150 字 (繁體中文) 簡短分析這位使用者的品味特質,以及為什麼這份推薦清單適合他。

使用者過往高評分作品: {likes}
推薦清單前 5 部: {titles}

請從具體面向 (例如:都偏黑暗心理 / 都有強女主 / 都節奏快 / 都重世界觀建構) 切入分析,不要泛泛而談。直接回覆,不要前言。"""
    return _chat(prompt, max_tokens=350, temperature=0.7)


def discovery_pitch(anime_title: str, anime_genre: str,
                    anime_synopsis: str, seed_context: str) -> str:
    """Feature B 探索卡片的單張作品副本 (~60 字)。"""
    if not is_configured():
        return ""
    prompt = f"""你是動漫推薦師。請用 1-2 句話 (60 字內,繁體中文) 介紹《{anime_title}》為什麼值得這位使用者考慮。

使用者偏好線索: {seed_context}
推薦動漫類型: {anime_genre}
劇情概要: {_shorten(anime_synopsis, 250)}

直接回覆,不要前言、不要重複問題。"""
    return _chat(prompt, max_tokens=200, temperature=0.85)
