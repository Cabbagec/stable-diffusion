import json
import logging
import uuid

from aiohttp import web

from diffusion_at_home.instance.app_server import BotServer
from diffusion_at_home.instance.job import JobStatus
from diffusion_at_home.instance.worker import Worker, WorkerStatus

routes = web.RouteTableDef()


def get_path(v):
    return f'/bot{v}'


@routes.post(get_path('/worker/{worker_id}/heartbeat'))
async def heartbeat(req: web.Request):
    """
    POST /worker/<worker_id>/heartbeat
    > request: Status
    {
        "status": "RUNNING/WAITING",
        # optional, running job id
        "job_id": <uuid>,
    }

    > response: {
        'abort': True/false
    }
    """
    req_data = await req.json()
    app: BotServer = req.app

    worker_id = req.match_info.get('worker_id')
    status = req_data.get('status')
    job_id = req_data.get('job_id')

    worker: Worker = app.get_worker(worker_id)
    if not worker:
        logging.info(f'New worker joined, id: {worker_id}')
        worker = app.add_worker(worker_id, status)

    worker.status = status
    logging.debug(f'heartbeat from {req.match_info["worker_id"]}, status: {status}')
    if status == WorkerStatus.WAITING and job_id:
        raise AssertionError(
            f'Worker {worker.worker_id}: is WAITING but claims job id {job_id}, ignoring'
        )

    if status == WorkerStatus.RUNNING:
        try:
            job = next(job for job in app.jobs_queue if job.job_id == job_id)
            if job.job_assignee is not worker:
                raise AssertionError(
                    f'Worker {worker.worker_id}: claims job {job.job_id}, but job does not mark worker as its assignee'
                )

            # if job.job_status == JobStatus.ABORT:
            #     logging.warning(f'Stopping job {job_id} on worker {worker_id}...')
            #     return web.json_response({'abort': True})

            if job.job_status != JobStatus.RUNNING:
                job.job_status = JobStatus.RUNNING

        except StopIteration:
            raise AssertionError(
                f'Worker {worker.worker_id}: claims job {job_id} but job of this id is already lost.'
            )

    return web.json_response({})


@routes.get(get_path('/worker/{worker_id}/job'))
async def get_job(req: web.Request):
    """
    /worker/{worker_id}/job
    < response:
    job.job_details
    {
        "job_id": <uuid>,
        "prompt": "",
        "width": 512,
        "height": 512,
        "steps": 50,
        "guidance_scale": 7,
        "seed": ...,
        "upscale": 4
    }
    or {
        "job_id": <uuid>,
        "resources": ['animation', 'upscalex2', 'upscalex3', 'upscalex4']
    }
    or {}
    """
    # req_data = await req.json()
    worker_id = req.match_info['worker_id']
    app: BotServer = req.app
    worker = app.get_worker(worker_id)
    # worker not registered yet, return empty
    if not worker:
        return web.json_response({})

    logging.info(f'Worker {worker.worker_id}: worker trying to get job')
    if worker.status != WorkerStatus.WAITING:
        logging.error(
            f'Worker {worker.worker_id}: worker is not WAITING but trying to get job'
        )
        return web.json_response({})

    # check if worker needs to fetch other resources
    if worker.resources_to_fetch:
        job_id, resources = worker.resources_to_fetch.popitem()
        res = {"job_id": job_id, 'resources': resources}
        return web.json_response(res)

    # find one assigned but still waiting job, resend this one
    try:
        assigned_not_running_job = next(
            job
            for job in app.jobs_queue
            if job.job_assignee is worker and job.job_status == JobStatus.WAITING
        )
        logging.info(
            f'Job {assigned_not_running_job.job_id} already assigned'
            f' to worker {worker.worker_id} but not started, resend'
        )
        return web.json_response(assigned_not_running_job.job_details)

    except StopIteration:
        pass

    # assigned, still running jobs by worker, but worker hasn't marked them as finished, and is requesting new one,
    # mark them as lost FAILED ones
    assigned_still_running_job = list(
        job
        for job in app.jobs_queue
        if job.job_assignee is worker and job.job_status == JobStatus.RUNNING
    )
    for _job in assigned_still_running_job:
        logging.info(
            f'Job {_job.job_id}: assigned to worker {worker.worker_id} is marked as lost FAILED'
        )
        _job.job_status = JobStatus.FAILED

    # peek one unassigned job from deque
    try:
        job = next(job for job in app.jobs_queue if not job.job_assignee)
    except StopIteration:
        logging.debug(
            f'Worker {worker.worker_id}: No job available, current job length: {len(app.jobs_queue)}'
        )
        return web.json_response({})

    # send details of selected job to current worker
    # res = web.json_response(job.job_details)
    # await res.prepare(req)
    # await res.write_eof()

    # confirm after sending successfully
    job.job_assignee = worker
    # send details of selected job to current worker
    return web.json_response(job.job_details)


@routes.post(get_path('/worker/{worker_id}/error'))
async def error(req: web.Request):
    """
    POST /worker/<worker_id>/heartbeat
    > request: Status
    {
        "job_id": <uuid>,
        "error": "xxx"
    }

    > response: {}
    """
    req_data = await req.json()
    app: BotServer = req.app

    worker_id = req.match_info.get('worker_id')
    job_id = req_data.get('job_id')
    job = app.get_job_by_id(job_id)
    if job:
        err = req_data.get('error', '')
        job.job_status = JobStatus.FAILED
        exec_callback(job.update_callback, msg=err)
        # app.jobs_queue.remove(job)
        return web.Response()


@routes.post(get_path('/worker/{worker_id}/resource'))
async def post_resource(req: web.Request):
    """
    /worer/{worker_id}/resource
        "job_id": ,
        "resource_type": "animation/Upscale x2/Upscale x3/Upscale x4"
        "resource": <file: xxx.xxx>
    """
    app: BotServer = req.app
    worker = app.get_worker(req.match_info['worker_id'])
    if not worker:
        logging.error(f'Unregistered worker uploading resources, rejecting...')
        return

    job_id = resource_type = filename = ''
    tmp_path = app.tmp_dir / str(uuid.uuid4())
    async for field in (await req.multipart()):
        if field.name == 'job_id':
            job_id = (await field.read()).decode()

        elif field.name == 'resource_type':
            resource_type = (await field.read()).decode()

        elif field.name == 'resource':
            filename = field.filename
            await app.write_field(field, path=tmp_path)

        else:
            logging.error(
                f'Unknown field {field.name} from worker {worker.worker_id} resources'
            )

    if job_id and resource_type and filename and tmp_path.exists():
        logging.info(
            f'Resources {resource_type} of job {job_id} received from worker {worker.worker_id}'
        )
        job = app.get_job_by_id(job_id)
        job.update_job_resource(resource_type, tmp_path, filename)

    else:
        logging.error(
            f'resources received, but some fields are empty.'
            f'job_id: {job_id}, resource_type: {resource_type}, filename: {filename}, worker_id: {worker.worker_id}'
        )


@routes.post(get_path('/worker/{worker_id}/report'))
async def report(request: web.Request):
    """
    /worer/{worker_id}/report
    < request: form-multipart
    "job_id": <uuid>,
    "index": 123
    # finished or not
    "completed": false/true
    "job_details": {
        "prompt": "...",
        "width": 512,
        "height": 512,
        "steps": 50,
        "guidance_scale": 4.0,
        "seed": xxx
    }

    "error": "xxx"

    file: <file: 123.png>
    result_img: <file: abc.png>
    result_gif: <file: abc.mp4>

    """
    app: BotServer = request.app
    worker = app.get_worker(request.match_info['worker_id'])

    # get all fields
    progress_tmp = progress_filename = job_id = index = result_img_tmp = None
    completed = False
    job_details = {}

    async for field in (await request.multipart()):
        if field.name == 'job_id':
            job_id = (await field.read()).decode()

        elif field.name == 'index':
            index = int(await field.read())

        elif field.name == 'file':
            progress_tmp = app.tmp_dir / str(uuid.uuid4())
            progress_filename = field.filename
            await app.write_field(field, progress_tmp)

        elif field.name == 'completed':
            completed = (await field.read()).decode().lower() == 'true'

        elif field.name == 'result_img':
            result_img_tmp = app.tmp_dir / str(uuid.uuid4())
            await app.write_field(field, result_img_tmp)
        #
        # elif field.name == 'result_gif':
        #     result_gif_tmp = app.tmp_dir / str(uuid.uuid4())
        #     await app.write_field(field, result_gif_tmp)

        elif field.name == 'job_details':
            job_details = json.loads((await field.read()).decode())

    # abort job if its status is aborting
    job = app.get_job_by_id(job_id, raise_exception=True)
    if job.job_status == JobStatus.ABORT:
        return web.json_response({'abort': True})

    # see if fields are complete, progress or result
    if None in (index, progress_tmp, progress_filename) and not result_img_tmp:
        logging.error(
            f'Worker {worker.worker_id}: report is not complete, '
            f'job_id: {job_id}, index: {index}, progress_tmp: {progress_tmp}, '
            f'result_img_tmp: {result_img_tmp}, '
            f'skipping'
        )
        return web.Response(status=500, reason='progress content incomplete')

    # update job details
    if not job_details or not isinstance(job_details, dict):
        logging.error(f'Worker {worker.worker_id} did not report job details')
        return web.Response(status=500, reason='progress content incomplete')

    job.job_details.update(job_details)

    # get progress filepath
    if not job.job_tmpdir:
        job.job_tmpdir = app.tmp_dir / str(job_id)
        if not job.job_tmpdir.exists():
            job.job_tmpdir.mkdir(parents=True)

    # job in progress, write progress, update status if possible
    if index is not None and not completed:
        if job.job_status != JobStatus.RUNNING:
            job.job_status = JobStatus.RUNNING
        if worker.status != WorkerStatus.RUNNING:
            worker.status = WorkerStatus.RUNNING

        name_ext = progress_filename.split('.')
        ext = name_ext[-1] if len(name_ext) == 2 else 'png'
        progress_filepath = job.job_tmpdir / f'{index:04d}.{ext}'
        progress_tmp.rename(progress_filepath)

        job.update_job_progress(index, progress_filepath.name)
        return web.json_response({})

    elif completed and result_img_tmp:
        # write results
        job.update_job_result(result_img_tmp)
        return web.json_response({})

    else:
        logging.error(
            f'Job {job_id}: Unknown report status, not progress nor completed, '
            f'completed: {completed}, index: {index}'
        )
        return web.Response(
            status=500, reason='Unknown report status, not progress nor completed'
        )
