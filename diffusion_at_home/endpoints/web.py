from aiohttp import web
from diffusion_at_home.config import web_client_tokens
from diffusion_at_home.instance.app_server import BotServer
from diffusion_at_home.instance.job import Job, JobStatus, JobSource
from diffusion_at_home.utils import filter_prompt

routes = web.RouteTableDef()


def get_web_endpoint(path):
    return f'/diffusion/web{path}'


def client_id_auth(req: web.Request):
    client_id = req.match_info.get('client_id')
    if not client_id or client_id not in web_client_tokens:
        return False

    return True


def error_response(msg: str):
    return web.json_response({'error': msg}, status=500)


@routes.post(get_web_endpoint('/{client_id}/create_job'))
async def create_job(req: web.Request):
    """
    Req: {
        "prompt": "...",
        "width": 512,
        "height": 512,
        "steps": 50,
        "guidance_scale": 7.0,
        "seed": 0
    }
    Res: {
        "job_id": <uuid>,
        "queue_len": 0,
        "prompt": "...",
        ...
    }
    """
    if not client_id_auth(req):
        return error_response('client not authenticated')

    # app: BotServer = req.app
    try:
        data = await req.json()
        prompt = filter_prompt(data.get('prompt'), raise_exception=False)
        if not prompt:
            return error_response('only english is allowed for prompt')

        width, height = data.get('width', 512), data.get('height', 512)
        if not width or not height:
            return error_response('please use correct dimension')

        width, height = int(width), int(height)
        if width % 64 != 0 or height % 64 != 0:
            return error_response(f'width and height must be multiples of 64')

        if width * height > 512 * 512 * 2.25:
            return error_response(
                'picture too large, please use smaller width or height'
            )

        steps = int(data.get('steps', 50))
        if not (0 < steps <= 100):
            return error_response(f'please use steps between 1 - 100')

    except Exception as e:
        return error_response(e.args[0])


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
