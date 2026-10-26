"""Прогон экстрактора по примерам из data/ и сравнение с эталонами.

python eval.py                          # отчёт в консоль
python eval.py --report docs/eval/v1.md # и сохранить его в файл
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.extractor import CardExtractor
from app.llm import GigaChatClient
from app.models import CrmCard, ExtractionResult

DATA = Path(__file__).resolve().parent / "data"

# Поле -> как его достать из карточки и как нормализовать перед сравнением
STRICT_FIELDS = {
    "company": lambda c: _plain(c.company),
    "contact.email": lambda c: _plain(c.contact.email),
    "contact.phone": lambda c: _phone(c.contact.phone),
    "urgency.level": lambda c: c.urgency.level,
    "urgency.deadline": lambda c: c.urgency.deadline,
    "products": lambda c: sorted(str(p.catalog_id) for p in c.products),
}
LOOSE_FIELDS = {
    "contact.name": lambda c: _plain(c.contact.name),
    "contact.position": lambda c: _plain(c.contact.position),
}


@dataclass
class EmailReport:
    name: str
    mismatches: dict[str, tuple] = field(default_factory=dict)  # поле -> (получили, ожидали)
    result: ExtractionResult | None = None
    expected: CrmCard | None = None
    error: str | None = None


def run_eval(extractor: CardExtractor) -> list[EmailReport]:
    reports = []
    for email_path in sorted((DATA / "emails").glob("*.txt")):
        report = EmailReport(name=email_path.stem)
        report.expected = CrmCard.model_validate_json((DATA / "expected" / f"{email_path.stem}.json").read_text(encoding="utf-8"))
        try:
            report.result = extractor.extract(email_path.read_text(encoding="utf-8"))
        except Exception as error:  # один упавший пример не должен ронять весь прогон
            report.error = f"{type(error).__name__}: {error}"
        else:
            for name, get in {**STRICT_FIELDS, **LOOSE_FIELDS}.items():
                got, want = get(report.result.card), get(report.expected)
                if got != want:
                    report.mismatches[name] = (got, want)
        reports.append(report)
    return reports


def render_report(reports: list[EmailReport], model: str) -> str:
    ok_reports = [r for r in reports if r.result]
    lines = [f"# Eval: {model}", "", "## Точность по полям", "", "| Поле | Верно |", "|---|---|"]
    for name in [*STRICT_FIELDS, *LOOSE_FIELDS]:
        correct = sum(name not in r.mismatches for r in ok_reports)
        strict = "" if name in STRICT_FIELDS else " (нестрого)"
        lines.append(f"| {name}{strict} | {correct}/{len(reports)} |")

    prompt_tokens = sum(r.result.prompt_tokens for r in ok_reports)
    completion_tokens = sum(r.result.completion_tokens for r in ok_reports)
    seconds = [r.result.seconds for r in ok_reports]
    lines += [
        "",
        "## Итого",
        "",
        f"- Писем без ошибок в строгих полях: {sum(not set(r.mismatches) & set(STRICT_FIELDS) for r in ok_reports)}/{len(reports)}",
        f"- Неподтверждённых цитатой полей: {sum(len(r.result.unconfirmed) for r in ok_reports)}",
        f"- Упавших вызовов: {len(reports) - len(ok_reports)}",
        f"- Токены: {prompt_tokens} на вход + {completion_tokens} на выход, в среднем {(prompt_tokens + completion_tokens) // max(len(ok_reports), 1)} на письмо",
        f"- Время: в среднем {sum(seconds) / max(len(seconds), 1):.1f} с, максимум {max(seconds, default=0):.1f} с",
        "",
        "## По письмам",
    ]

    for r in reports:
        lines += ["", f"### {r.name}", ""]
        if r.error:
            lines.append(f"- ❌ Ошибка: {r.error}")
            continue
        if not r.mismatches:
            lines.append("- ✅ Все поля совпали с эталоном")
        for name, (got, want) in r.mismatches.items():
            lines.append(f"- ❌ {name}: получили `{got}`, ожидали `{want}`")
        if r.result.unconfirmed:
            lines.append(f"- ⚠️ Не подтверждены цитатой: {', '.join(r.result.unconfirmed)}")
        lines.append(f"- Суть (модель): {r.result.card.request_summary}")
        lines.append(f"- Суть (эталон): {r.expected.request_summary}")
        if r.result.warnings:
            lines.append(f"- Предупреждения модели: {'; '.join(r.result.warnings)}")
    return "\n".join(lines) + "\n"


def _plain(value: str | None) -> str | None:
    """Для сравнения без учёта регистра, кавычек и лишних пробелов."""
    if value is None:
        return None
    value = re.sub(r"[«»“”\"']", "", value.lower().replace("ё", "е"))
    return re.sub(r"\s+", " ", value).strip()


def _phone(value: str | None) -> str | None:
    """Только цифры; российские 8XXXXXXXXXX и +7XXXXXXXXXX — один номер."""
    digits = re.sub(r"\D", "", value or "")
    if len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    return digits or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, help="куда сохранить отчёт в markdown")
    args = parser.parse_args()

    llm = GigaChatClient.from_env()
    reports = run_eval(CardExtractor(llm))
    report = render_report(reports, llm.model)
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        # Полные ответы модели рядом с отчётом — пригодятся при разборе ошибок
        raw = {r.name: r.result.model_dump(mode="json") if r.result else {"error": r.error} for r in reports}
        args.report.with_suffix(".json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Отчёт сохранён в {args.report} и {args.report.with_suffix('.json')}")


if __name__ == "__main__":
    main()
