from openai import AzureOpenAI
import base64
import requests
import time

class ChatGPT:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer <OPENAI_KEY>'
    }
    def __init__(self) -> None:
        self.max_try = 100

    def get_response_text(self, system_message, user_message, schema):
        if schema is None:
            data = {
                "model": "gpt-4o-2024-08-06",
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
            }
        else:
            data = {
                "model": "gpt-4o-2024-08-06",
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": schema
                }
            }
        for i in range(self.max_try):
            try:
                response = requests.post(self.url, headers=self.headers, json=data)
                result = response.json()['choices'][0]['message']['content']
                return result
            except Exception as e:
                print(e)
                time.sleep(2)
                print(f"Retry {i}")
                continue


if __name__ == "__main__":
    pass