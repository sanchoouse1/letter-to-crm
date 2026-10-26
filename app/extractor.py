import json
import re
import sys
from pathlib import Path

from pydantic import ValidationError

from app.llm import GigaChatClient
from app.models import CrmCard, ExtractionResult, LLMAnswer

ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = ROOT / "prompts" / "extract_card.md"
CATALOG_PATH = ROOT / "data" / "products.json"


class CardExtractor:
    """Письмо -> карточка CRM: промпт, вызов модели, разбор ответа и проверка цитат."""

    def __init__(self, llm: GigaChatClient, prompt_path: Path = PROMPT_PATH, catalog_path: Path = CATALOG_PATH):
        self._llm = llm
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        self._system_prompt = prompt_path.read_text(encoding="utf-8").replace("{catalog}", format_catalog(catalog))

    def extract(self, email: str) -> ExtractionResult:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": email},
        ]
        response = self._llm.chat(messages)
        prompt_tokens, completion_tokens, seconds = response.prompt_tokens, response.completion_tokens, response.seconds

        try:
            answer = parse_answer(response.text)
        except (ValueError, ValidationError) as error:
            # Один повтор: показываем модели её ответ и ошибку валидации
            messages += [
                {"role": "assistant", "content": response.text},
                {"role": "user", "content": f"Ответ не прошёл проверку: {error}\nВерни исправленный JSON целиком."},
            ]
            response = self._llm.chat(messages)
            prompt_tokens += response.prompt_tokens
            completion_tokens += response.completion_tokens
            seconds += response.seconds
            answer = parse_answer(response.text)

        return ExtractionResult(
            card=answer.card,
            evidence=answer.evidence,
            warnings=answer.warnings,
            unconfirmed=find_unconfirmed(answer.card, answer.evidence, email),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            seconds=round(seconds, 2),
        )


def format_catalog(catalog: list[dict]) -> str:
    return "\n".join(f"- {item['id']} — {item['name']} (синонимы: {', '.join(item['aliases'])})" for item in catalog)


def parse_answer(text: str) -> LLMAnswer:
    """Достаёт JSON из ответа, даже если модель обернула его в ```json или добавила текст вокруг."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("в ответе нет JSON-объекта")
    return LLMAnswer.model_validate_json(text[start : end + 1])


def find_unconfirmed(card: CrmCard, evidence: dict[str, str], email: str) -> list[str]:
    """Поля, у которых нет цитаты или цитата не встречается в письме, — кандидаты в галлюцинации."""
    text = normalize(email)
    return [
        path
        for path in filled_fields(card)
        if not evidence.get(path) or normalize(evidence[path]) not in text
    ]


def filled_fields(card: CrmCard) -> list[str]:
    values = {
        "company": card.company,
        "contact.name": card.contact.name,
        "contact.position": card.contact.position,
        "contact.email": card.contact.email,
        "contact.phone": card.contact.phone,
        "request_summary": card.request_summary,
        "urgency.level": card.urgency.level,
        "urgency.deadline": card.urgency.deadline,
    }
    paths = [path for path, value in values.items() if value is not None]
    return paths + [f"products[{i}]" for i in range(len(card.products))]


def normalize(text: str) -> str:
    text = re.sub(r"^[ \t]*>+[ \t]?", "", text, flags=re.MULTILINE)  # маркеры цитирования
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[«»“”„\"]", '"', text)
    text = re.sub(r"[–—]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    # python -m app.extractor data/emails/01_simple_kp.txt
    email_text = Path(sys.argv[1]).read_text(encoding="utf-8")
    result = CardExtractor(GigaChatClient.from_env()).extract(email_text)
    print(result.model_dump_json(indent=2))
