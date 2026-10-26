import os
import time
import uuid
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from openai import OpenAI

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
API_URL = "https://gigachat.devices.sberbank.ru/api/v1"
# Токен живёт 30 минут; обновляем заранее, чтобы не поймать протухший посреди запроса
TOKEN_REFRESH_MARGIN_SEC = 60


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class GigaChatClient:
    """GigaChat через OpenAI-совместимый API. Сам получает и обновляет OAuth-токен."""

    def __init__(self, auth_key: str, scope: str, model: str, ca_bundle: str | bool = True):
        self.model = model
        self._auth_key = auth_key
        self._scope = scope
        self._http = httpx.Client(verify=ca_bundle, timeout=60)
        self._openai = OpenAI(base_url=API_URL, api_key="pending", http_client=self._http)
        self._token: str | None = None
        self._token_expires_at = 0.0

    @classmethod
    def from_env(cls) -> "GigaChatClient":
        load_dotenv()
        return cls(
            auth_key=os.environ["GIGACHAT_AUTH_KEY"],
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            model=os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max"),
            ca_bundle=os.getenv("GIGACHAT_CA_BUNDLE") or True,
        )

    def chat(self, messages: list[dict], temperature: float = 0.1) -> LLMResponse:
        started = time.perf_counter()
        completion = self._openai.with_options(api_key=self._get_token()).chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return LLMResponse(
            text=completion.choices[0].message.content or "",
            prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
            completion_tokens=completion.usage.completion_tokens if completion.usage else 0,
            seconds=round(time.perf_counter() - started, 2),
        )

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_expires_at - TOKEN_REFRESH_MARGIN_SEC:
            return self._token
        response = self._http.post(
            OAUTH_URL,
            headers={"Authorization": f"Basic {self._auth_key}", "RqUID": str(uuid.uuid4())},
            data={"scope": self._scope},
        )
        response.raise_for_status()
        payload = response.json()
        self._token = payload["access_token"]
        self._token_expires_at = payload["expires_at"] / 1000  # GigaChat отдаёт миллисекунды
        return self._token


if __name__ == "__main__":
    # Smoke-проверка: python -m app.llm
    client = GigaChatClient.from_env()
    reply = client.chat([{"role": "user", "content": "Ответь одним словом: связь есть?"}])
    print(f"{client.model}: {reply.text!r} | {reply.prompt_tokens}+{reply.completion_tokens} токенов | {reply.seconds} с")
