import uuid

from aiohttp import web

from diffusion_at_home.config import web_client_tokens, size_limit
from diffusion_at_home.instance.app_server import BotServer, ServerJobWaitingException
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
    return web.json_response({'error': msg or 'Unknown error'}, status=500)


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
        client_id = req.match_info['client_id']
        data = await req.json()
        app: BotServer = req.app
        prompt = filter_prompt(data.get('prompt'), raise_exception=False)
        if not prompt:
            return error_response('only english is allowed for prompt')

        width, height = data.get('width', 512), data.get('height', 512)
        if not width or not height:
            return error_response('please use correct dimension')

        width, height = int(width), int(height)
        if width % 64 != 0 or height % 64 != 0:
            return error_response(f'width and height must be multiples of 64')

        if width * height > size_limit:
            return error_response(
                'picture too large, please use smaller width or height'
            )

        steps = int(data.get('steps', 50))
        if not (0 < steps <= 100):
            return error_response(f'please use steps between 1 - 100')

        guidance_scale = float(data.get('guidance_scale') or data.get('scale') or 4.0)
        guidance_scale = min(100.0, max(0.0, guidance_scale))

        seed = int(data.get('seed', uuid.uuid4().int & ((1 << 32) - 1)))
        seed = min(4294967295, max(0, seed))

        job_details = {
            'prompt': prompt,
            'width': width,
            'height': height,
            'guidance_scale': guidance_scale,
            'steps': steps,
            'scale': guidance_scale,
            'seed': seed,
        }

        job = Job(JobSource.WEB, job_details=job_details, client_id=client_id)
        app.add_job(job)
        return web.json_response({'job_id': job.job_id, 'queue_len': 0, **job_details})

    except ServerJobWaitingException as e:
        return web.json_response(
            {'job_id': job.job_id, **job_details, 'queue_len': e.args[1]}
        )

    except Exception as e:
        return error_response(e.args[0])


@routes.get(get_web_endpoint('/{client_id}/get_job/{job_id}'))
async def get_job_status(req: web.Request):
    if not client_id_auth(req):
        return error_response('client not authenticated')

    try:
        app: BotServer = req.app
        job_id = req.match_info['job_id']
        client_id = req.match_info['client_id']
        job = app.get_job_by_id(job_id, raise_exception=True)
        assert JobSource.WEB == job.job_source and client_id == job.client_id

        if job.job_status not in (JobStatus.RUNNING, JobStatus.FINISHED):
            raise Exception(f'job isn\'t running or finished')

        if job.job_progress:
            finished = False
            if job.job_result:
                finished = True
            return web.json_response(
                {'index': max(job.job_progress), 'finished': finished}
            )

        raise Exception(f'No progress yet. Wait for a few seconds please.')

    except Exception as e:
        return error_response(e.args[0])


@routes.get(get_web_endpoint('/{client_id}/get_job/{job_id}/{filename}'))
async def get_job_file(req: web.Request):
    if not client_id_auth(req):
        return error_response('client not authenticated')

    try:
        app: BotServer = req.app
        client_id = req.match_info['client_id']
        job_id = req.match_info['job_id']
        filename = req.match_info['filename']
        job = app.get_job_by_id(job_id, raise_exception=True)
        assert JobSource.WEB == job.job_source and client_id == job.client_id

        if filename in job.job_progress.values() or filename in job.job_result.values():
            return web.FileResponse(job.job_tmpdir / filename)

        raise Exception(f'Cannot find file {filename} under job {job_id}')

    except Exception as e:
        return error_response(e.args[0])


@routes.post(get_web_endpoint('/{client_id}/abort_job/{job_id}'))
async def abort_job(req: web.Request):
    if not client_id_auth(req):
        return error_response('client not authenticated')

    return error_response('Not implemented')


@routes.post(get_web_endpoint('/{client_id}/upscale_job/{job_id}'))
async def upscale(req: web.Request):
    if not client_id_auth(req):
        return error_response('client not authenticated')

    return error_response('Not implemented')


@routes.post(get_web_endpoint('/{client_id}/merge_imgs/{job_id}'))
async def merge_imgs(req: web.Request):
    if not client_id_auth(req):
        return error_response('client not authenticated')

    return error_response('Not implemented')
