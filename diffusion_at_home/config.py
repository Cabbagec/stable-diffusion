tg_bot_token = 'token'

# White lists
allowed_chat_ids = [123]

# web clients
web_client_tokens = ['test_client']

allowed_commands = [
    {
        'command': 'prompt',
        'description': 'Start generate with the prompt, English only. 使用提示语开始生成，只可用英语。',
    }
]

# others
name = 'HomeDiffusionBot'
cache_dir = '/tmp/diffusionbot'
telegram_bot_api_server = f'https://api.telegram.org/bot{tg_bot_token}/'

size_limit = 512 * 512 * 2.25