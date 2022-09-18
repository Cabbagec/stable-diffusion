import asyncio
import logging
import re

from diffusion_at_home.config import telegram_bot_api_server


def get_tg_endpoint(name):
    return f'{telegram_bot_api_server}{name}'


def exec_callback(callback, *args, **kwargs):
    if callback:
        if asyncio.iscoroutinefunction(callback):
            asyncio.create_task(callback(*args, **kwargs))
        elif callable(callback):
            callback(*args, **kwargs)


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


def filter_prompt(text: str):
    if re.match(r'^[0-9a-zA-Z,\.\s!]+$', text):
        return text
    else:
        raise Exception('English only please')
