import os
import asyncio
import html
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.filters import CommandStart, Command

from src.search import SearchEngine
from src.rag import generate_answer
from src.logger import logger

load_dotenv()
API_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not API_TOKEN or API_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
    logger.warning("TELEGRAM_BOT_TOKEN is not set in .env or is using placeholder!")

bot = Bot(token=API_TOKEN if API_TOKEN else "")
dp = Dispatcher()

# Instantiate the SearchEngine
search_engine = SearchEngine()

class SearchStates(StatesGroup):
    waiting_for_strict = State()
    waiting_for_semantic = State()
    waiting_for_combined = State()
    waiting_for_rag = State()

# Persistent reply keyboard for main menu
def get_main_keyboard():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="🔍 Строгий поиск"),
                KeyboardButton(text="🧠 Семантический поиск")
            ],
            [
                KeyboardButton(text="🚀 Комбинированный поиск"),
                KeyboardButton(text="🤖 ИИ-Ассистент")
            ],
            [
                KeyboardButton(text="📅 Инсайт дня")
            ]
        ],
        resize_keyboard=True,
        placeholder="Выберите действие..."
    )
    return keyboard

def get_help_text() -> str:
    return (
        "📖 <b>Как пользоваться поиском:</b>\n\n"
        "🔎 <b>1. Точный поиск (по конкретным словам)</b>\n"
        "Ищет точное совпадение слов или фраз, прозвучавших в лекциях. Полезно, если ты помнишь конкретное слово.\n"
        "• <b>Примеры:</b> <code>борщ</code>, <code>выгорание в декрете</code>, <code>бывш[ая|ей]</code>\n"
        "• <b>Выход:</b> Точные цитаты с таймкодами на видео.\n\n"
        "🧠 <b>2. Поиск по смыслу (умный поиск)</b>\n"
        "Ищет фрагменты, близкие по значению к твоему запросу, даже если слова не совпадают.\n"
        "• <b>Примеры:</b> <i>как пережить разрыв отношений</i>, <i>почему девушка манипулирует</i>\n"
        "• <b>Выход:</b> Разборы близких тем с таймкодами.\n\n"
        "🚀 <b>3. Комбинированный поиск (слова + смысл)</b>\n"
        "Сначала находит ключевые слова, а затем отбирает из них самые близкие по значению.\n"
        "• <b>Пример:</b> <i>бывшая вернулась после игнора</i>\n"
        "• <b>Выход:</b> Сбалансированные фрагменты по теме.\n\n"
        "🤖 <b>4. ИИ-Ассистент (RAG)</b>\n"
        "Синтезирует подробный ответ на твой вопрос на основе материалов лекций с указанием источников и таймкодов.\n"
        "• <b>Пример:</b> <i>почему игнор работает?</i>\n"
        "• <b>Запуск:</b> Нажми 🤖 ИИ-Ассистент в меню или напиши /ask\n\n"
        "📅 <b>5. Инсайт дня</b>\n"
        "Случайный полезный тезис или саммари из лекции с таймкодом для быстрого обучения.\n"
        "• <b>Запуск:</b> Нажми 📅 Инсайт дня в меню или напиши /daily\n\n"
        "👇 Выбери режим в меню или введи команду!"
    )

@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    user_id = message.from_user.id
    username = message.from_user.username or "unknown"
    logger.info(f"User {user_id} (@{username}) triggered /start.")
    await state.clear()
    welcome_text = (
        "👋 <b>Привет! Я твой ассистент по материалам Максима Вердикта.</b>\n"
        "Я помогу тебе найти нужные мысли, цитаты и ответы на вопросы по психологии отношений прямо из сотен его видеолекций.\n\n"
    )
    await message.answer(welcome_text + get_help_text(), parse_mode="HTML", reply_markup=get_main_keyboard())

@dp.message(Command("help"))
async def cmd_help(message: Message, state: FSMContext):
    user_id = message.from_user.id
    logger.info(f"User {user_id} triggered /help.")
    await state.clear()
    await message.answer(get_help_text(), parse_mode="HTML", reply_markup=get_main_keyboard())

@dp.message(Command("daily"))
@dp.message(F.text == "📅 Инсайт дня")
async def process_daily_command(message: Message, state: FSMContext):
    user_id = message.from_user.id
    logger.info(f"User {user_id} requested daily insight.")
    await state.clear()
    await message.answer("🔄 <i>Ищу инсайт для тебя...</i>", parse_mode="HTML")
    insight_data = await asyncio.to_thread(search_engine.get_random_insight)
    if not insight_data:
        logger.warning(f"Daily insight requested by {user_id} could not be fulfilled: database contains no summaries/key points.")
        await message.answer(
            "😔 К сожалению, в базе пока нет сохраненных инсайтов/тезисов. "
            "Пожалуйста, запустите процесс генерации саммари в Colab/Kaggle и импортируйте их.",
            reply_markup=get_main_keyboard()
        )
        return

    chunk, insight = insight_data
    video_title = escape_html(chunk.video.title)
    source_name = escape_html(chunk.source)
    formatted_time = format_time(chunk.start_time)
    
    yt_url = chunk.video.url
    if chunk.start_time is not None:
        yt_url += f"&t={int(chunk.start_time)}"
        
    response_text = (
        f"📅 <b>Инсайт дня из материалов лекций:</b>\n\n"
        f"💡 <i>«{escape_html(insight)}»</i>\n\n"
        f"🎥 <b>Лекция:</b> <a href='{yt_url}'>{video_title}</a>\n"
        f"📌 <i>Раздел: {source_name}</i> (⏱ {formatted_time})"
    )
    logger.info(f"Delivering daily insight from chunk ID {chunk.id} to user {user_id}.")
    await message.answer(response_text, parse_mode="HTML", reply_markup=get_main_keyboard(), disable_web_page_preview=True)

@dp.message(Command("ask"))
@dp.message(F.text == "🤖 ИИ-Ассистент")
async def process_rag_mode(message: Message, state: FSMContext):
    user_id = message.from_user.id
    logger.info(f"User {user_id} entered RAG prompt mode.")
    await state.set_state(SearchStates.waiting_for_rag)
    await message.answer(
        "🤖 <b>Режим ИИ-Ассистента (RAG)</b>\n"
        "Задай мне любой вопрос о психологии отношений. Я найду релевантные фрагменты лекций и составлю для тебя подробный ответ со ссылками на видео и таймкоды.\n\n"
        "✍️ Отправь свой вопрос:",
        parse_mode="HTML"
    )

@dp.message(F.text == "🔍 Строгий поиск")
async def process_strict_mode(message: Message, state: FSMContext):
    user_id = message.from_user.id
    logger.info(f"User {user_id} selected Strict Search mode.")
    await state.set_state(SearchStates.waiting_for_strict)
    await message.answer(
        "✍️ Отправь мне слово или фразу для точного поиска:",
        parse_mode="HTML"
    )

@dp.message(F.text == "🧠 Семантический поиск")
async def process_semantic_mode(message: Message, state: FSMContext):
    user_id = message.from_user.id
    logger.info(f"User {user_id} selected Semantic Search mode.")
    await state.set_state(SearchStates.waiting_for_semantic)
    await message.answer(
        "✍️ Отправь мне свой вопрос или тему для поиска по смыслу:",
        parse_mode="HTML"
    )

@dp.message(F.text == "🚀 Комбинированный поиск")
async def process_combined_mode(message: Message, state: FSMContext):
    user_id = message.from_user.id
    logger.info(f"User {user_id} selected Combined Search mode.")
    await state.set_state(SearchStates.waiting_for_combined)
    await message.answer(
        "✍️ Отправь мне свой запрос для комбинированного поиска:",
        parse_mode="HTML"
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

def escape_html(text: str) -> str:
    return html.escape(text) if text else ""

def highlight_strict_query(text: str, query: str) -> str:
    if not query:
        return text
    import re
    escaped_query = html.escape(query)
    try:
        pattern = rf"(&[a-zA-Z0-9#]+;)|({escaped_query})"
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        escaped_literal = re.escape(escaped_query)
        pattern = rf"(&[a-zA-Z0-9#]+;)|({escaped_literal})"
        rx = re.compile(pattern, re.IGNORECASE)

    def replace(match):
        if match.group(1):
            return match.group(1)
        return f"<u><b>{match.group(0)}</b></u>"

    return rx.sub(replace, text)

async def perform_search_and_respond(message: Message, state: FSMContext, search_type: str, query: str):
    user_id = message.from_user.id
    logger.info(f"Processing {search_type} search for User {user_id}. Query: '{query}'")
    await message.answer("🔄 <i>Ищу подходящие фрагменты...</i>", parse_mode="HTML")
    
    # Run database and embedding computations inside thread pool to prevent event loop blocking
    try:
        if search_type == "strict":
            results = await asyncio.to_thread(search_engine.strict_search, query, limit=3)
        elif search_type == "semantic":
            results = await asyncio.to_thread(search_engine.semantic_search, query, limit=3)
        else:
            results = await asyncio.to_thread(search_engine.combined_search, query, limit=3)
            
        logger.info(f"Search engine returned {len(results)} matches for User {user_id}.")
        
        if not results:
            await message.answer(
                "❌ Ничего не найдено по вашему запросу. Попробуйте сформулировать иначе.",
                reply_markup=get_main_keyboard()
            )
            await state.clear()
            return
            
        # Send header message
        await message.answer(f"✨ <b>Результаты поиска ({len(results)}):</b>", parse_mode="HTML")
        
        for idx, (chunk, score) in enumerate(results, 1):
            video_title = escape_html(chunk.video.title)
            source_name = escape_html(chunk.source)
            chunk_text = escape_html(chunk.text)
            formatted_time = format_time(chunk.start_time)
            
            # Create YouTube URL with timestamp parameter
            yt_url = chunk.video.url
            if chunk.start_time:
                yt_url += f"&t={int(chunk.start_time)}"
                
            base_text = (
                f"{idx}. 🎥 <b><a href='{yt_url}'>{video_title}</a></b>\n"
                f"📌 <i>Раздел: {source_name}</i> (⏱ {formatted_time})\n"
                f"<blockquote>{{}}</blockquote>"
            )
            
            # Telegram character limit is 4096. Restrict chunk size based on template length.
            max_chunk_len = max(100, 4090 - len(base_text))
            if len(chunk_text) > max_chunk_len:
                chunk_text = chunk_text[:max_chunk_len - 3] + "..."
            
            # Highlight query terms for strict search
            if search_type == "strict":
                chunk_text = highlight_strict_query(chunk_text, query)
                
            result_text = base_text.format(chunk_text)
            
            # Attach keyboard menu on the final message to restore custom menu state
            is_last = (idx == len(results))
            reply_markup = get_main_keyboard() if is_last else None
            
            await message.answer(
                result_text, 
                parse_mode="HTML", 
                reply_markup=reply_markup, 
                disable_web_page_preview=True
            )
    except Exception as e:
        logger.error(f"Error during bot search for query '{query}': {e}", exc_info=True)
        await message.answer("⚠️ Произошла ошибка во время поиска. Пожалуйста, попробуйте позже.")
        
    await state.clear()

@dp.message(SearchStates.waiting_for_strict)
async def process_strict_query(message: Message, state: FSMContext):
    await perform_search_and_respond(message, state, "strict", message.text)

@dp.message(SearchStates.waiting_for_semantic)
async def process_semantic_query(message: Message, state: FSMContext):
    await perform_search_and_respond(message, state, "semantic", message.text)

@dp.message(SearchStates.waiting_for_combined)
async def process_combined_query(message: Message, state: FSMContext):
    await perform_search_and_respond(message, state, "combined", message.text)

@dp.message(SearchStates.waiting_for_rag)
async def process_rag_query(message: Message, state: FSMContext):
    query = message.text
    user_id = message.from_user.id
    logger.info(f"Processing RAG assistant query for User {user_id}: '{query}'")
    
    await message.answer("🔄 <i>ИИ-Ассистент думает над ответом... Это может занять до 15-20 секунд.</i>", parse_mode="HTML")
    
    try:
        # Retrieve context chunks
        results = await asyncio.to_thread(search_engine.combined_search, query, limit=4)
        logger.info(f"Retrieved {len(results)} chunks for RAG synthesis context.")
        
        if not results:
            logger.warning(f"RAG search query '{query}' by User {user_id} returned no relevant chunks.")
            await message.answer(
                "❌ Не удалось найти подходящие материалы в лекциях для ответа на этот вопрос. Попробуйте сформулировать иначе.",
                reply_markup=get_main_keyboard()
            )
            await state.clear()
            return
            
        chunks = [chunk for chunk, _ in results]
        
        # Generate synthesized answer via OpenRouter
        answer = await asyncio.to_thread(generate_answer, query, chunks)
        
        # Split and send if answer is too long for Telegram (4096 chars limit)
        if len(answer) > 4000:
            parts = [answer[i:i+4000] for i in range(0, len(answer), 4000)]
            logger.info(f"Splitting RAG response for User {user_id} into {len(parts)} message segments.")
            for idx, part in enumerate(parts):
                reply_markup = get_main_keyboard() if idx == len(parts) - 1 else None
                await message.answer(
                    part, 
                    parse_mode="HTML", 
                    reply_markup=reply_markup, 
                    disable_web_page_preview=True
                )
        else:
            logger.info(f"Sending single RAG response (length {len(answer)}) to User {user_id}.")
            await message.answer(
                answer, 
                parse_mode="HTML", 
                reply_markup=get_main_keyboard(), 
                disable_web_page_preview=True
            )
            
    except Exception as e:
        logger.error(f"Error in RAG process: {e}", exc_info=True)
        await message.answer(
            "⚠️ Произошла ошибка во время генерации ответа от ИИ. Пожалуйста, попробуйте позже.",
            reply_markup=get_main_keyboard()
        )
        
    await state.clear()

# Catch-all handler for regular messages
@dp.message()
async def process_unknown_message(message: Message):
    user_id = message.from_user.id
    logger.info(f"User {user_id} sent unknown message content: '{message.text}'")
    await message.answer(
        "💡 Пожалуйста, выберите режим в меню ниже:",
        reply_markup=get_main_keyboard()
    )

async def main():
    logger.info("Starting Telegram Bot dispatcher...")
    try:
        await dp.start_polling(bot)
    finally:
        logger.info("Stopping Telegram Bot polling and cleaning resources.")
        search_engine.close()

if __name__ == "__main__":
    if not API_TOKEN or API_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        logger.critical("ERROR: Please set a valid TELEGRAM_BOT_TOKEN in your .env file before running!")
    else:
        asyncio.run(main())
