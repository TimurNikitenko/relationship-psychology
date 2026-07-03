import os
import re
import requests
from dotenv import load_dotenv

from src.logger import logger

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = (
    "Ты — профессиональный ИИ-ассистент по материалам лекций психолога Максима Вердикта.\n"
    "Твоя задача — ответить на вопрос пользователя, используя исключительно предоставленные текстовые фрагменты его видеолекций.\n\n"
    "Правила:\n"
    "1. Отвечай подробно, структурированно и только на основе предоставленного контекста. Не придумывай информацию от себя.\n"
    "2. В ответе обязательно ссылайся на первоисточники (видео). У каждого фрагмента есть название видео, базовая ссылка и таймкод начала.\n"
    "   Оформляй ссылки в виде кликабельного HTML-текста, например: <a href='{URL}&t={SECONDS}'>{Название видео} (⏱ {Таймкод})</a>.\n"
    "3. Если предоставленный контекст не содержит ответа на вопрос, вежливо ответь, что в имеющихся материалах лекций Максима Вердикта нет информации по этому вопросу.\n"
    "4. Отвечай строго на русском языке. Используй красивую HTML-разметку, разрешенную в Telegram (<b>жирный</b> для акцентов, <i>курсив</i>, <u>подчеркивание</u>, <blockquote>цитаты</blockquote>, <code>код</code>, ссылки <a href='URL'>текст</a>).\n"
    "   КРИТИЧЕСКИ ВАЖНО: Telegram НЕ поддерживает HTML-теги списков <ul>, <ol>, <li>, абзацев <p> и переносов строк <br>. Не используй их ни в коем случае! Для списков используй символ • или дефис (-), для абзацев и переносов используй обычный перевод строки (символ \\n).\n"
)

def format_time(seconds: float) -> str:
    if seconds is None:
        return "00:00"
    s = int(seconds)
    hours = s // 3600
    minutes = (s % 3600) // 60
    secs = s % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def sanitize_telegram_html(text: str) -> str:
    """Translates and strips HTML tags to conform with Telegram's strict HTML subset parser."""
    if not text:
        return ""
    # Translate list items to bullet points
    text = text.replace("<li>", "• ").replace("</li>", "\n")
    # Translate paragraph endings to newlines
    text = text.replace("</p>", "\n\n")
    # Translate br tags to newlines
    text = re.sub(r'<br\s*/?>', '\n', text)
    # Strip any tags NOT supported by Telegram's HTML parser
    supported_tag_pattern = r'<(?!/?(?:a|b|i|u|s|code|pre|blockquote|strong|em|ins|strike|del|tg-spoiler|span)\b)[^>]+>'
    text = re.sub(supported_tag_pattern, '', text)
    # Clean up double line breaks/spaces
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def generate_answer(query: str, chunks: list) -> str:
    logger.info(f"RAG Request - Query: '{query}' with {len(chunks)} context chunks.")
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        logger.error("RAG Request failed: OPENROUTER_API_KEY is not configured.")
        return (
            "⚠️ <b>Ошибка конфигурации ИИ-Ассистента:</b>\n"
            "Не задан API-ключ <code>OPENROUTER_API_KEY</code> в файле .env. "
            "Пожалуйста, настройте интеграцию с OpenRouter для использования этой функции."
        )

    # Construct context text
    context_items = []
    for idx, chunk in enumerate(chunks, 1):
        formatted_t = format_time(chunk.start_time)
        yt_url = chunk.video.url
        if chunk.start_time is not None:
            yt_url += f"&t={int(chunk.start_time)}"
        
        logger.debug(f"  Context Chunk {idx}: Video ID {chunk.video.video_id}, Timestamp: {formatted_t}")
        
        item_text = (
            f"--- ФРАГМЕНТ {idx} ---\n"
            f"Видео: {chunk.video.title}\n"
            f"Ссылка: {yt_url}\n"
            f"Таймкод: {formatted_t}\n"
            f"Текст фрагмента: {chunk.text}\n"
        )
        context_items.append(item_text)

    context_str = "\n".join(context_items)

    user_content = (
        f"КОНТЕКСТ ИЗ ЛЕКЦИЙ:\n{context_str}\n\n"
        f"ВОПРОС ПОЛЬЗОВАТЕЛЯ: {query}\n\n"
        f"Напиши ответ на основе контекста выше, соблюдая правила."
    )

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/timur/relationships-psychology",
        "X-Title": "Relationships Psychology Assistant",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.3
    }

    try:
        logger.info(f"Sending prompt to OpenRouter API (Model: {OPENROUTER_MODEL})")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            raw_answer = data["choices"][0]["message"]["content"]
            logger.info("OpenRouter response received successfully.")
            logger.debug(f"Raw response: {raw_answer}")
            
            sanitized = sanitize_telegram_html(raw_answer)
            logger.debug(f"Sanitized response: {sanitized}")
            return sanitized
        else:
            logger.error(f"OpenRouter API response did not contain Choices. Response body: {data}")
            return "⚠️ Не удалось получить ответ от ИИ-ассистента. Некорректный формат ответа API."
            
    except requests.exceptions.HTTPError as he:
        logger.error(f"OpenRouter HTTP Error: {he}", exc_info=True)
        try:
            err_msg = response.json()
            return f"⚠️ <b>Ошибка API OpenRouter:</b> {err_msg.get('error', {}).get('message', he)}"
        except Exception:
            return f"⚠️ <b>Ошибка API OpenRouter:</b> HTTP status {response.status_code}"
    except Exception as e:
        logger.error(f"RAG Error: {e}", exc_info=True)
        return f"⚠️ Произошла ошибка при обращении к ИИ-ассистенту: {e}"
