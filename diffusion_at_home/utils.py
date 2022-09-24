import asyncio
import logging
import re

from diffusion_at_home.config import telegram_bot_api_server


def get_tg_endpoint(name):
    return f'{telegram_bot_api_server}{name}'


backgroud_tasks = set()


def exec_callback(callback, *args, **kwargs):
    if callback:
        if asyncio.iscoroutinefunction(callback):
            task = asyncio.create_task(callback(*args, **kwargs))
            backgroud_tasks.add(task)
            task.add_done_callback(backgroud_tasks.discard)
        elif callable(callback):
            callback(*args, **kwargs)


def exec_callback_chain(*callbacks):
    async def chain():
        for callback in callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback()

            elif callable(callback):
                callback()

            else:
                logging.error(
                    f'Unknown callback: {callback}, not a coroutine function nor a function'
                )

    exec_callback(chain)


def get_param(data: dict, params, default=None):
    if not isinstance(data, dict):
        return data if not params else default

    if not params:
        return data

    if isinstance(params, (str, int, float)):
        return data.get(params, default)

    next_data = data.get(params[0])
    if not next_data:
        return default
    return get_param(next_data, params[1:], default)


def filter_prompt(text: str, raise_exception=True):
    if not text:
        if raise_exception:
            raise Exception(f'Text is empty')
        return

    if re.match(r'^[0-9a-zA-Z,\.\s\-&!]+$', text):
        return text
    elif raise_exception:
        raise Exception('English only please')
    else:
        return
