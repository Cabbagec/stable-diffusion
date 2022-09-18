import logging

from aiohttp import web

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s]:%(levelname)s: %(message)s'
)


async def init():
    from diffusion_at_home.instance.app_server import BotServer
    from diffusion_at_home.endpoints.telegram import routes as tg_routes
    from diffusion_at_home.endpoints.worker import routes as worker_routes

    app = BotServer()
    app.add_routes(tg_routes)
    app.add_routes(worker_routes)
    await app.start_bot_session()
    return app


def main():
    web.run_app(init(), host='127.0.0.1', port=1999)
    logging.info(f'Bye...')


if __name__ == '__main__':
    main()
