"""Тонкая HTTP-обёртка над ядром (заглушка под сервис). Запуск: uvicorn lawyer.api:app

Это скелет, а не готовый продакшен: нет авторизации, очереди задач и хранилища.
Список обязательного перед выходом к клиентам — docs/ROADMAP.md.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from .bootstrap import build_deps
from .config import Settings
from .deps import Deps
from .review import ReviewPaths, run_review
from .schemas import ClientProfile

MAX_FILE_BYTES = 10_000_000
ALLOWED_SUFFIXES = {".pdf", ".xlsx", ".xlsm", ".csv", ".txt", ".xml"}


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

        review_id = uuid.uuid4().hex
        case_dir = settings.root / "out" / "api_cases" / review_id
        (case_dir / "docs").mkdir(parents=True)
        (case_dir / "profile.yaml").write_text(
            yaml.safe_dump(parsed.model_dump(mode="json"), allow_unicode=True), encoding="utf-8"
        )
        for upload in files:
            name = Path(upload.filename or "").name  # режем путь: защита от ../
            if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
                raise HTTPException(status_code=415, detail=f"формат не поддерживается: {name}")
            data = await upload.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise HTTPException(status_code=413, detail=f"файл больше 10 МБ: {name}")
            (case_dir / "docs" / name).write_bytes(data)

        result = await run_review(shared, case_dir, paths, ask=None)
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
