"""Tests for the CLI ingestion script."""

from unittest.mock import MagicMock, patch

import pytest

from scripts.ingest import find_files, main


class TestFindFiles:
    def test_returns_single_pdf(self, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"fake")
        assert find_files(pdf) == [pdf]

    def test_rejects_unsupported_extension(self, tmp_path):
        csv = tmp_path / "notes.csv"
        csv.write_bytes(b"a,b,c")
        with pytest.raises(SystemExit):
            find_files(csv)

    def test_returns_single_txt(self, tmp_path):
        txt = tmp_path / "notes.txt"
        txt.write_bytes(b"text")
        assert find_files(txt) == [txt]

    def test_finds_supported_files_in_directory(self, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"a")
        (tmp_path / "b.pdf").write_bytes(b"b")
        (tmp_path / "c.txt").write_bytes(b"c")
        result = find_files(tmp_path)
        assert len(result) == 3
        assert {f.suffix for f in result} == {".pdf", ".txt"}

    def test_finds_pdfs_recursively(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "top.pdf").write_bytes(b"t")
        (sub / "nested.pdf").write_bytes(b"n")
        result = find_files(tmp_path)
        assert len(result) == 2

    def test_exits_when_no_supported_files_in_dir(self, tmp_path):
        (tmp_path / "notes.csv").write_bytes(b"a,b,c")
        with pytest.raises(SystemExit):
            find_files(tmp_path)

    def test_exits_for_nonexistent_path(self, tmp_path):
        missing = tmp_path / "nope.pdf"
        with pytest.raises(SystemExit):
            find_files(missing)


class TestMain:
    @patch("scripts.ingest.get_settings")
    def test_exits_without_args(self, mock_get_settings, monkeypatch):
        monkeypatch.setattr("sys.argv", ["ingest.py"])
        with pytest.raises(SystemExit):
            main()

    @patch("scripts.ingest.ingest_document")
    @patch("scripts.ingest.DocumentStore")
    @patch("scripts.ingest.get_settings")
    def test_ingests_single_file(
        self, mock_get_settings, mock_store_cls, mock_ingest, tmp_path, monkeypatch
    ):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake-pdf")

        settings = MagicMock()
        settings.rag_enabled = True
        mock_get_settings.return_value = settings
        mock_ingest.return_value = {
            "doc_id": "abc123",
            "filename": "test.pdf",
            "chunks_stored": 5,
        }

        monkeypatch.setattr("sys.argv", ["ingest.py", str(pdf)])
        main()

        mock_ingest.assert_called_once_with(
            b"fake-pdf",
            "test.pdf",
            mock_store_cls.return_value,
            settings,
            extra_metadata={},
        )

    @patch("scripts.ingest.get_settings")
    def test_exits_when_rag_disabled(self, mock_get_settings, tmp_path, monkeypatch):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake")

        settings = MagicMock()
        settings.rag_enabled = False
        mock_get_settings.return_value = settings

        monkeypatch.setattr("sys.argv", ["ingest.py", str(pdf)])
        with pytest.raises(SystemExit):
            main()

    @patch("scripts.ingest.ingest_document")
    @patch("scripts.ingest.DocumentStore")
    @patch("scripts.ingest.get_settings")
    def test_continues_on_failure(
        self,
        mock_get_settings,
        mock_store_cls,
        mock_ingest,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        (tmp_path / "good.pdf").write_bytes(b"ok")
        (tmp_path / "bad.pdf").write_bytes(b"nope")

        settings = MagicMock()
        settings.rag_enabled = True
        mock_get_settings.return_value = settings

        mock_ingest.side_effect = [
            {"doc_id": "a1", "filename": "bad.pdf", "chunks_stored": 3},
            ValueError("parse error"),
        ]

        monkeypatch.setattr("sys.argv", ["ingest.py", str(tmp_path)])
        main()

        output = capsys.readouterr().out
        assert "1 succeeded" in output
        assert "1 failed" in output

    @patch("scripts.ingest.ingest_document")
    @patch("scripts.ingest.DocumentStore")
    @patch("scripts.ingest.get_settings")
    def test_clear_flag_deletes_before_ingesting(
        self,
        mock_get_settings,
        mock_store_cls,
        mock_ingest,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake-pdf")

        settings = MagicMock()
        settings.rag_enabled = True
        mock_get_settings.return_value = settings

        mock_store = mock_store_cls.return_value
        mock_store.clear_all.return_value = 15

        mock_ingest.return_value = {
            "doc_id": "abc123",
            "filename": "test.pdf",
            "chunks_stored": 5,
        }

        monkeypatch.setattr("sys.argv", ["ingest.py", str(pdf), "--clear"])
        main()

        mock_store.clear_all.assert_called_once()
        mock_ingest.assert_called_once()
        output = capsys.readouterr().out
        assert "Cleared 15 existing chunks" in output
