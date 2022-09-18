from aiohttp import web

routes = web.RouteTableDef()


def get_web_endpoint(path):
    return f'/diffusion/web{path}'


@routes.post(get_web_endpoint('/{client_id}/create_job'))
async def create_job(req: web.Request):
    pass


@routes.get(get_web_endpoint('/{client_id}/get_job/{job_id}'))
async def get_job_status(req: web.Request):
    pass


@routes.post(get_web_endpoint('/{client_id}/abort_job/{job_id}'))
async def abort_job(req: web.Request):
    pass


@routes.post(get_web_endpoint('/{client_id}/upscale_job/{job_id}'))
async def upscale(req: web.Request):
    pass


@routes.post(get_web_endpoint('/{client_id}/merge_imgs/{job_id}'))
async def merge_imgs(req: web.Request):
    pass
