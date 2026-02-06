import os
import time

import requests
from loguru import logger


class ChatGPT:
    url = "https://api.openai.com/v1/chat/completions"

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it before running: export OPENAI_API_KEY='your-key'"
            )
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.max_try = 100

    def get_response_text(self, system_message, user_message, schema):
        if schema is None:
            data = {
                "model": "gpt-4o-2024-08-06",
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
            }
        else:
            data = {
                "model": "gpt-4o-2024-08-06",
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                "response_format": {"type": "json_schema", "json_schema": schema},
            }
        for i in range(self.max_try):
            try:
                response = requests.post(self.url, headers=self.headers, json=data)
                result = response.json()["choices"][0]["message"]["content"]
                return result
            except Exception as e:
                logger.warning(f"API request failed: {e}")
                time.sleep(2)
                logger.info(f"Retry {i}")
                continue
