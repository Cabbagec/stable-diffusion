import logging

from aiohttp import web

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s]:%(levelname)s: %(message)s'
)


async def init():
    app = BotServer()
    app.add_routes(routes)
    await app.start_bot_session()
    return app


def main():
    web.run_app(init(), host='127.0.0.1', port=1999)
    logging.info(f'Bye...')


if __name__ == '__main__':
    main()
