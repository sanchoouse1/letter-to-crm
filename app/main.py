"""uvicorn app.main:app --reload  ->  http://localhost:8000"""

import json
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.extractor import CATALOG_PATH, ROOT, CardExtractor
from app.llm import GigaChatClient
from app.models import ExtractionResult

STATIC = ROOT / "app" / "static"
EMAILS = ROOT / "data" / "emails"

app = FastAPI(title="Письмо → карточка CRM")


class ExtractRequest(BaseModel):
    email: str = Field(min_length=10, description="Текст письма вместе с заголовками От / Дата / Тема")


@lru_cache
def get_extractor() -> CardExtractor:
    # Создаём при первом запросе: страница и примеры открываются и без настроенного ключа
    return CardExtractor(GigaChatClient.from_env())


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/examples")
def examples() -> list[dict]:
    return [{"name": path.stem, "text": path.read_text(encoding="utf-8")} for path in sorted(EMAILS.glob("*.txt"))]


@app.get("/api/products")
def products() -> list[dict]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


@app.post("/api/extract")
def extract(request: ExtractRequest) -> ExtractionResult:
    try:
        return get_extractor().extract(request.email)
    except Exception as error:  # для черновика достаточно показать причину в UI
        raise HTTPException(status_code=502, detail=f"Не удалось извлечь карточку: {type(error).__name__}: {error}")
