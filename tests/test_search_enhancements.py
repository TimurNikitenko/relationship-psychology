import unittest
import os
import re
import sqlalchemy as sa
from unittest.mock import MagicMock, patch

from src.database import Chunk, Video
from src.bot import highlight_strict_query
from src.search import SearchEngine

class TestSearchEnhancements(unittest.TestCase):
    def test_highlight_strict_query_basic(self):
        text = "Она была бывшей, а теперь она просто знакомая."
        # Match case-insensitive
        result = highlight_strict_query(text, "бывш")
        self.assertIn("<u><b>бывш</b></u>ей", result)

    def test_highlight_strict_query_regex(self):
        text = "Она была бывшей, а теперь она бывшая."
        # Correct regex alternation (ая or ей)
        result = highlight_strict_query(text, "бывш(ая|ей)")
        self.assertIn("<u><b>бывшей</b></u>", result)
        self.assertIn("<u><b>бывшая</b></u>", result)

    def test_highlight_strict_query_html_entity_safety(self):
        # The query is "amp", text has escaped & as &amp;
        # Highlighting should not touch the &amp; entity
        text = "Алекс &amp; Барбара обсуждали отношения."
        result = highlight_strict_query(text, "amp")
        # Check that we did not split the HTML entity
        self.assertEqual(result, text)
        
        # If text has "amp" elsewhere, it should still be highlighted
        text2 = "Алекс &amp; Барбара на приеме у амплитудного терапевта."
        result2 = highlight_strict_query(text2, "амп")
        self.assertIn("<u><b>амп</b></u>литудного", result2)
        self.assertIn("&amp;", result2)  # HTML entity remains intact

    def test_highlight_strict_query_invalid_regex(self):
        # Invalid regex (unclosed parenthesis)
        text = "Она была бывшей."
        result = highlight_strict_query(text, "бывш(")
        # Should gracefully fallback to literal matching and highlight
        self.assertIn("<u><b>бывш(</b></u>", result) if "бывш(" in text else self.assertEqual(result, text)

    @patch("src.search.SessionLocal")
    @patch("src.search.SearchEngine.load_bm25_corpus")
    def test_search_engine_fallback_trigger(self, mock_load, mock_session):
        # Test that fallback_search is triggered when there are no matches
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        
        engine = SearchEngine()
        engine.bm25 = MagicMock()
        engine.bm25.get_scores.return_value = []
        
        # Mock semantic search raw result to be empty
        mock_db.query.return_value.options.return_value.order_by.return_value.limit.return_value.all.return_value = []
        
        # Setup fallback mock
        mock_chunk = MagicMock(spec=Chunk)
        mock_chunk.text = "Hello world"
        mock_chunk.id = 1
        
        with patch.object(engine, "fallback_search", return_value=[(mock_chunk, 1.0)]) as mock_fallback:
            results = engine.combined_search("тест", limit=5)
            # It should call fallback because RRF scores is empty
            mock_fallback.assert_called_once()
            self.assertEqual(len(results), 1)

    @patch("src.search.SessionLocal")
    @patch("src.search.SearchEngine.load_bm25_corpus")
    def test_get_random_insight_no_data(self, mock_load, mock_session):
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        
        # When DB has no data, return empty list
        mock_db.query.return_value.options.return_value.filter.return_value.filter.return_value.all.return_value = []
        mock_db.query.return_value.options.return_value.filter.return_value.all.return_value = []
        
        engine = SearchEngine()
        insight = engine.get_random_insight()
        self.assertIsNone(insight)

    @patch("src.search.SessionLocal")
    @patch("src.search.SearchEngine.load_bm25_corpus")
    def test_get_random_insight_with_data(self, mock_load, mock_session):
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        
        video = MagicMock(spec=Video)
        video.title = "Test Video"
        video.url = "https://youtube.com/watch?v=123"
        
        chunk = MagicMock(spec=Chunk)
        chunk.video = video
        chunk.source = "Part 1"
        chunk.start_time = 100.0
        chunk.key_points = ["Уважение — залог баланса значимости."]
        
        # Mock DB returning our chunk
        mock_db.query.return_value.options.return_value.filter.return_value.filter.return_value.all.return_value = [chunk]
        
        engine = SearchEngine()
        result = engine.get_random_insight()
        
        self.assertIsNotNone(result)
        res_chunk, res_insight = result
        self.assertEqual(res_chunk, chunk)
        self.assertEqual(res_insight, "Уважение — залог баланса значимости.")

    def test_sanitize_telegram_html(self):
        from src.rag import sanitize_telegram_html
        raw_text = (
            "Привет! Вот список:\n"
            "<ul>"
            "<li>Пункт 1</li>"
            "<li>Пункт 2</li>"
            "</ul>"
            "<p>Абзац с <b>жирным</b> и <i>курсивом</i>.</p>"
            "<div>Удаляемый тег, но текст остается.</div>"
            "<a href='http://test.com'>Ссылка</a>"
        )
        cleaned = sanitize_telegram_html(raw_text)
        # Check that ul and ol tags are removed
        self.assertNotIn("<ul>", cleaned)
        self.assertNotIn("</ul>", cleaned)
        # Check that li tags are converted to bullets
        self.assertIn("• Пункт 1", cleaned)
        # Check that p and div tags are removed but content remains
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("<div>", cleaned)
        self.assertIn("Удаляемый тег, но текст остается.", cleaned)
        # Check that bold, italic, and links are kept
        self.assertIn("<b>жирным</b>", cleaned)
        self.assertIn("<i>курсивом</i>", cleaned)
        self.assertIn("<a href='http://test.com'>Ссылка</a>", cleaned)

if __name__ == "__main__":
    unittest.main()
