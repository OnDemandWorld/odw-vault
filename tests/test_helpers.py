"""Tests for pipeline/helpers.py."""

import json

import sqlite_utils

from pipeline.helpers import (
    classify_archive_error,
    is_archive_extension,
    is_hidden_or_system,
    is_system_dir,
    now_iso,
    record_expansion,
    record_failure,
    sanitize_json_for_sqlite,
    sha256_file,
)


class TestSha256File:
    def test_known_content(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello", encoding="utf-8")
        path, digest, error = sha256_file(str(f))
        assert error is None
        assert digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_missing_file(self):
        path, digest, error = sha256_file("/nonexistent/file.txt")
        assert digest is None
        assert error is not None
        assert "No such file" in error or "cannot open" in error.lower()

    def test_binary_content(self, tmp_path):
        f = tmp_path / "bin.dat"
        f.write_bytes(b"\x00\x01\x02\x03\xff")
        _, digest, error = sha256_file(str(f))
        assert error is None
        assert len(digest) == 64


class TestIsHiddenOrSystem:
    def test_hidden_dot_file(self):
        assert is_hidden_or_system(".DS_Store") is True
        assert is_hidden_or_system(".localized") is True

    def test_non_hidden(self):
        assert is_hidden_or_system("readme.txt") is False
        assert is_hidden_or_system("document.pdf") is False

    def test_hidden_but_not_system(self):
        assert is_hidden_or_system(".gitignore") is True
        assert is_hidden_or_system(".env") is True
        assert is_hidden_or_system(".git/config") is True

    def test_starts_with_dot_system(self):
        assert is_hidden_or_system(".Trashes") is True
        assert is_hidden_or_system(".Spotlight-V100") is True


class TestIsSystemDir:
    def test_macos_dir(self):
        assert is_system_dir("__MACOSX") is True
        assert is_system_dir(".AppleDouble") is True

    def test_normal_dir(self):
        assert is_system_dir("docs") is False
        assert is_system_dir("project") is False


class TestIsArchiveExtension:
    def test_archive_ext(self):
        assert is_archive_extension(".zip") is True
        assert is_archive_extension(".rar") is True
        assert is_archive_extension(".7z") is True
        assert is_archive_extension(".tar.gz") is True
        assert is_archive_extension(".tar") is True

    def test_doc_archive_not_expanded(self):
        assert is_archive_extension(".docx") is False
        assert is_archive_extension(".xlsx") is False
        assert is_archive_extension(".pptx") is False
        assert is_archive_extension(".pages") is False
        assert is_archive_extension(".epub") is False

    def test_non_archive(self):
        assert is_archive_extension(".txt") is False
        assert is_archive_extension(".pdf") is False
        assert is_archive_extension(".json") is False

    def test_case_insensitive(self):
        assert is_archive_extension(".ZIP") is True
        assert is_archive_extension(".Zip") is True


class TestClassifyArchiveError:
    def test_encrypted(self):
        assert classify_archive_error(Exception("Archive is encrypted")) == "encrypted"
        assert classify_archive_error(Exception("Password required")) == "encrypted"

    def test_corrupt(self):
        assert classify_archive_error(Exception("Corrupt archive")) == "corrupt"
        assert classify_archive_error(Exception("Invalid archive")) == "corrupt"
        assert classify_archive_error(Exception("Damaged file")) == "corrupt"

    def test_permission(self):
        assert classify_archive_error(Exception("Permission denied")) == "permission"

    def test_timeout(self):
        assert classify_archive_error(Exception("Operation timeout")) == "timeout"

    def test_unknown(self):
        assert classify_archive_error(Exception("Something went wrong")) == "unknown"


class TestNowIso:
    def test_returns_iso_format(self):
        result = now_iso()
        assert result.endswith("Z")
        assert "T" in result

    def test_parsable(self):
        from datetime import datetime

        result = now_iso()
        # Should be parseable
        datetime.fromisoformat(result.replace("Z", "+00:00"))


class TestSanitizeJsonForSqlite:
    def test_basic_dict(self):
        result = sanitize_json_for_sqlite({"key": "value"})
        assert json.loads(result) == {"key": "value"}

    def test_unicode(self):
        result = sanitize_json_for_sqlite({"text": "中文"})
        assert "中文" in result

    def test_default_str(self):
        from datetime import datetime

        result = sanitize_json_for_sqlite({"ts": datetime(2026, 1, 1, 0, 0, 0)})
        assert "2026" in result


class TestRecordFailure:
    def test_inserts_row(self, tmp_path):
        db = sqlite_utils.Database(str(tmp_path / "test.db"))
        db["failure"].create(
            {
                "id": int,
                "file_id": int,
                "folder_id": int,
                "phase": str,
                "tool": str,
                "error_class": str,
                "error_message": str,
                "traceback": str,
                "occurred_at": str,
            }
        )
        record_failure(
            db, phase="test", tool="test_tool", error_class="test_error", error_message="test msg"
        )
        rows = list(db["failure"].rows)
        assert len(rows) == 1
        assert rows[0]["phase"] == "test"
        assert rows[0]["error_message"] == "test msg"


class TestRecordExpansion:
    def test_inserts_row(self, tmp_path):
        db = sqlite_utils.Database(str(tmp_path / "test.db"))
        db["archive_expansion"].create(
            {
                "id": int,
                "archive_file_id": int,
                "extracted_to_path": str,
                "extracted_to_folder_id": int,
                "tool": str,
                "succeeded": int,
                "file_count": int,
                "error_message": str,
                "extracted_at": str,
            }
        )
        record_expansion(
            db,
            archive_file_id=1,
            extracted_to_path="/tmp/out",
            extracted_to_folder_id=None,
            tool="patool",
            succeeded=True,
            file_count=5,
            error_message=None,
        )
        rows = list(db["archive_expansion"].rows)
        assert len(rows) == 1
        assert rows[0]["archive_file_id"] == 1
        assert rows[0]["succeeded"] == 1
