"""Тонкая HTTP-обёртка над ядром (заглушка под сервис). Запуск: uvicorn lawyer.api:app

Это скелет, а не готовый продакшен: нет авторизации, очереди задач и хранилища.
Список обязательного перед выходом к клиентам — docs/ROADMAP.md.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .bootstrap import build_deps
from .config import Settings
from .deps import Deps
from .review import ReviewPaths, run_review
from .schemas import ClientProfile

MAX_FILE_BYTES = 10_000_000
MAX_TOTAL_BYTES = 30_000_000
MAX_FILES = 10
MAX_BODY_BYTES = MAX_TOTAL_BYTES + 1_000_000
ALLOWED_SUFFIXES = {".pdf", ".xlsx", ".xlsm", ".csv", ".txt", ".xml"}

logger = logging.getLogger(__name__)


def _purge(case_dir: Path, out_dir: Path, review_id: str) -> None:
    """Удаляет документы и отчёт кейса. Сбой очистки логируем: молча оставить ПДн нельзя."""
    try:
        shutil.rmtree(case_dir, ignore_errors=False)
    except FileNotFoundError:
        pass
    except OSError:
        logger.exception("не удалось удалить каталог кейса %s", review_id)
    for suffix in ("_report.md", "_findings.json"):
        try:
            (out_dir / f"{review_id}{suffix}").unlink(missing_ok=True)
        except OSError:
            logger.exception("не удалось удалить результат кейса %s", review_id)


def create_app(deps: Deps | None = None) -> FastAPI:
    settings = deps.settings if deps else Settings.from_env()
    shared = deps or build_deps(settings)
    app = FastAPI(title="ai-lawyer", version="0.1.0")
    paths = ReviewPaths(
        knowledge_root=settings.root / "knowledge",
        rules_file=settings.root / "rules" / "usn.yaml",
        checklist_file=settings.root / "skills" / "usn-marketplace-review" / "checklist.md",
        out_dir=settings.root / "out" / "review",
    )

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        # Отсекает заведомо огромные запросы до разбора формы. Chunked-загрузки без
        # Content-Length это не ловит: лимит тела нужен и на уровне прокси.
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "тело запроса слишком большое"}, status_code=413)
        return await call_next(request)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/reviews")
    async def create_review(
        profile: str = Form(..., description="JSON ClientProfile"),
        files: list[UploadFile] = File(...),
    ) -> dict[str, object]:
        try:
            parsed = ClientProfile.model_validate(json.loads(profile))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=f"profile: {exc}") from exc

        if len(files) > MAX_FILES:
            raise HTTPException(status_code=413, detail=f"не больше {MAX_FILES} файлов за запрос")

        # Сначала валидируем и читаем всё, потом пишем на диск: отказ не оставляет мусора.
        # Имена файлов серверные: без коллизий, без ../ и без 300-символьных имён.
        uploads: list[tuple[str, bytes]] = []
        total = 0
        for i, upload in enumerate(files, 1):
            original = Path(upload.filename or "")
            if original.suffix.lower() not in ALLOWED_SUFFIXES:
                raise HTTPException(
                    status_code=415, detail=f"формат не поддерживается: {original.name[:80]}"
                )
            data = await upload.read(MAX_FILE_BYTES + 1)
            total += len(data)
            if len(data) > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                raise HTTPException(status_code=413, detail="файл или набор файлов слишком большой")
            stem = re.sub(r"[^\w\-]+", "_", original.stem)[:40] or "doc"
            uploads.append((f"{i:02d}_{stem}{original.suffix.lower()}", data))

        review_id = uuid.uuid4().hex
        case_dir = settings.root / "out" / "api_cases" / review_id
        try:
            (case_dir / "docs").mkdir(parents=True)
            (case_dir / "profile.yaml").write_text(
                yaml.safe_dump(parsed.model_dump(mode="json"), allow_unicode=True),
                encoding="utf-8",
            )
            for name, data in uploads:
                (case_dir / "docs" / name).write_bytes(data)
            result = await run_review(shared, case_dir, paths, ask=None)
        finally:
            _purge(case_dir, paths.out_dir, review_id)  # документы клиента (ПДн) не храним
        return {
            "id": review_id,
            "open_questions": result.open_questions,
            "findings": [f.model_dump(mode="json") for f in result.findings],
            "report_md": result.report_md,
        }

    return app


def __getattr__(name: str) -> FastAPI:
    # ленивое создание: импорт модуля не должен требовать ключей
    if name == "app":
        return create_app()
    raise AttributeError(name)
