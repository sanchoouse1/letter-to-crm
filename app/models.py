from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

UrgencyLevel = Literal["high", "medium", "low"]


class Contact(BaseModel):
    name: str | None = None
    position: str | None = None
    email: str | None = None
    phone: str | None = None


class ProductItem(BaseModel):
    catalog_id: str | None = Field(None, description="ID из справочника или null, если не нашлось")
    name_in_email: str = Field(description="Как продукт назван в письме")
    quantity: int | None = None


class Urgency(BaseModel):
    level: UrgencyLevel | None = None
    deadline: date | None = None
    reason: str | None = None


class CrmCard(BaseModel):
    """Карточка в том виде, в каком она ложится в CRM."""

    company: str | None = None
    contact: Contact = Contact()
    request_summary: str | None = None
    products: list[ProductItem] = []
    urgency: Urgency = Urgency()


class LLMAnswer(BaseModel):
    """Ответ модели: карточка плюс цитаты-обоснования и предупреждения."""

    card: CrmCard
    evidence: dict[str, str] = Field(
        default_factory=dict,
        description="Путь к полю (company, contact.email, urgency.deadline, ...) -> дословная цитата из письма",
    )
    warnings: list[str] = []


class ExtractionResult(BaseModel):
    """Что отдаём наружу: ответ модели, проверка цитат и стоимость вызова."""

    card: CrmCard
    evidence: dict[str, str]
    warnings: list[str]
    unconfirmed: list[str] = Field(description="Поля, цитату для которых не удалось найти в письме")
    prompt_tokens: int
    completion_tokens: int
    seconds: float
