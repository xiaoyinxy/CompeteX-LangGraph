import asyncio
import json
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


async def main() -> None:
    load_dotenv()
    model = init_chat_model(
        "deepseek:deepseek-v4-flash",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        max_tokens=32,
    )
    response = await model.ainvoke("只回复：连接成功")
    print(json.dumps({"ok": True, "has_content": bool(response.content)}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
