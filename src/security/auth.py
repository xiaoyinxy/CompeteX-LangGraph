import os
import asyncio
from langgraph_sdk import Auth
from langgraph_sdk.auth.types import StudioUser
from supabase import create_client, Client
from typing import Optional, Any

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Optional[Client] = None

if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)

# `Auth` 对象是 LangGraph 用于注册身份验证函数的容器
auth = Auth()


# `authenticate` 装饰器要求 LangGraph 将此函数作为每个请求的中间件，
# 用于判断是否允许对应请求
@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    """使用 Supabase 检查用户 JWT Token 是否有效。"""

    # 确认请求包含 Authorization Header
    if not authorization:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="缺少 Authorization Header"
        )

    # 解析 Authorization Header
    try:
        scheme, token = authorization.split()
        assert scheme.lower() == "bearer"
    except (ValueError, AssertionError):
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Authorization Header 格式无效"
        )

    # 确认 Supabase Client 已初始化
    if not supabase:
        raise Auth.exceptions.HTTPException(
            status_code=500, detail="Supabase Client 尚未初始化"
        )

    try:
        # 通过 asyncio.to_thread 在独立线程中使用 Supabase 解码并验证 JWT Token，避免阻塞
        async def verify_token() -> dict[str, Any]:
            response = await asyncio.to_thread(supabase.auth.get_user, token)
            return response

        response = await verify_token()
        user = response.user

        if not user:
            raise Auth.exceptions.HTTPException(
                status_code=401, detail="Token 无效或未找到用户"
            )

        # 验证通过后返回用户信息
        return {
            "identity": user.id,
        }
    except Exception as e:
        # 处理 Supabase 返回的异常
        raise Auth.exceptions.HTTPException(
            status_code=401, detail=f"身份验证失败：{str(e)}"
        )


@auth.on.threads.create
@auth.on.threads.create_run
async def on_thread_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.create.value,
):
    """创建 Thread 时添加 owner。

    此处理器在创建新 Thread 时执行两项操作：
    1. 在新建 Thread 的 metadata 中记录所有权；
    2. 返回过滤条件，确保只有创建者可以访问。
    """

    if isinstance(ctx.user, StudioUser):
        return

    # 为新建 Thread 添加 owner metadata，并随 Thread 持久保存
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity


@auth.on.threads.read
@auth.on.threads.delete
@auth.on.threads.update
@auth.on.threads.search
async def on_thread_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.read.value,
):
    """只允许用户读取自己的 Thread。

    此处理器在读取操作时执行。由于 Thread 已经存在，无需再次设置 metadata，
    只需返回过滤条件，确保用户只能看到自己的 Thread。
    """
    if isinstance(ctx.user, StudioUser):
        return

    return {"owner": ctx.user.identity}


@auth.on.assistants.create
async def on_assistants_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.create.value,
):
    if isinstance(ctx.user, StudioUser):
        return

    # 为新建 Assistant 添加 owner metadata，并随 Assistant 持久保存
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity


@auth.on.assistants.read
@auth.on.assistants.delete
@auth.on.assistants.update
@auth.on.assistants.search
async def on_assistants_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.read.value,
):
    """只允许用户读取自己的 Assistant。

    此处理器在读取操作时执行。由于 Assistant 已经存在，无需再次设置 metadata，
    只需返回过滤条件，确保用户只能看到自己的 Assistant。
    """

    if isinstance(ctx.user, StudioUser):
        return

    return {"owner": ctx.user.identity}


@auth.on.store()
async def authorize_store(ctx: Auth.types.AuthContext, value: dict):
    if isinstance(ctx.user, StudioUser):
        return

    # 每个 Store 项目的 `namespace` 字段都是一个元组，可以将其理解为项目所在目录
    namespace: tuple = value["namespace"]
    assert namespace[0] == ctx.user.identity, "无权执行此操作"
