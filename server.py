import asyncio
import json
import logging
import re
import ssl
import uuid
from collections import deque
from pathlib import Path, PurePath
from typing import Dict, Deque

import aiohttp
from aiohttp import web, BodyPartReader

name = 'HomeDiffusionBot'
tg_bot_token = "token"
cache_dir = '/tmp/diffusionbot'

allowed_chat_ids = [123]
allowed_commands = [
    {
        'command': 'prompt',
        'description': 'Start generate with the prompt, English only. 使用提示语开始生成，只可用英语。',
    }
]

logging.basicConfig(
    level=logging.DEBUG, format='[%(asctime)s]:%(levelname)s: %(message)s'
)

telegram_bot_api_server = f'https://api.telegram.org/bot{tg_bot_token}/'


def get_tg_endpoint(name):
    return f'{telegram_bot_api_server}{name}'


def dump_json_to_file(dict_json: dict, file):
    with open(file, 'w+') as f:
        json.dump(dict_json, ensure_ascii=False, indent=4, fp=f)


def get_client_session():
    ssl_context = ssl.create_default_context(cafile='/etc/ssl/cert.pem')
    conn = aiohttp.TCPConnector(ssl_context=ssl_context)
    return aiohttp.ClientSession(trust_env=True, connector=conn)


def exec_callback(callback, *args, **kwargs):
    if callback:
        if asyncio.iscoroutinefunction(callback):
            asyncio.create_task(callback(*args, **kwargs))
        elif callable(callback):
            callback(*args, **kwargs)


class BotServer(web.Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # worker_id: Worker instance
        self.workers: Dict[str, Worker] = dict()
        self.jobs_queue: Deque[Job] = deque()
        self.jobs_updated_map = {}
        self.jobs_finished_map = {}

        _cache_dir = Path(cache_dir)
        if _cache_dir.exists():
            if _cache_dir.is_file():
                raise Exception(f'Path {cache_dir} exists and is not a directory')
        else:
            _cache_dir.mkdir(parents=True)
        self.tmp_dir = _cache_dir

        # TODO: rate limit from telegram
        self.rate_limits = {'TELEGRAM': 0}
        self.telegram_session = None

    def add_job(self, job: 'Job', update_callback=None, finish_callback=None):
        workers = self.get_workers_by_status(WorkerStatus.WAITING)
        if not workers:
            available_workers = len(self.get_workers_by_status(WorkerStatus.RUNNING))
            if available_workers:
                exec_callback(
                    update_callback,
                    msg=f'Waiting... {available_workers} worker(s) available currently.',
                )

            else:
                logging.info(f'No workers available for job {job.job_id}, abandon')
                exec_callback(
                    update_callback,
                    msg=f'Sorry 😢, no workers are available currently. '
                    f'Please try again when workers are available.',
                )
                job.job_status = JobStatus.FAILED
                return

        job.update_callback = update_callback
        job.finish_callback = finish_callback
        self.jobs_queue.append(job)
        exec_callback(update_callback, msg='🤩 Job added, waiting for generating...')
        logging.info(f'New job added: {job.job_id}, detail: {job.job_details}')

    def add_worker(self, worker_id, status=None, timeout=10):
        worker = Worker(worker_id, status=status, timeout=timeout)
        self.workers[worker_id] = worker
        return worker

    def get_worker(self, worker_id):
        worker = self.workers.get(worker_id)
        if not worker:
            logging.error(f'Cannot find worker by id: {worker_id}')
        return worker

    def get_workers_by_status(self, status):
        if status not in (
            WorkerStatus.WAITING,
            WorkerStatus.RUNNING,
            WorkerStatus.OFFLINE,
        ):
            raise Exception(
                f'Unknown status {status}, wont find any worker by this status'
            )
        return [worker for _, worker in self.workers.items() if worker.status == status]

    def get_job_by_id(self, job_id, raise_exception=False):
        try:
            return next(job for job in self.jobs_queue if job.job_id == job_id)
        except StopIteration:
            logging.info(f'Cannot find job by id: {job_id}')
            if raise_exception:
                raise
            return

    def destroy_worker(self, worker_id):
        self.workers.pop(worker_id, None)

    async def write_field(self, field: BodyPartReader, path: PurePath, in_bytes=True):
        path = Path(path)
        mode = 'w+'
        if in_bytes:
            mode = 'w+b'
        with open(path, mode) as f:
            if in_bytes:
                while True:
                    chunk = await field.read_chunk()
                    if not chunk:
                        break
                    f.write(chunk)
            else:
                while True:
                    chunk = (await field.read_chunk()).decode()
                    if not chunk:
                        break
                    f.write(chunk)

    async def callback_loop(self):
        while True:
            await asyncio.sleep(3)
            while self.jobs_updated_map:
                job_id, job = self.jobs_updated_map.popitem()
                logging.info(f'Job {job_id}: calling update callback')
                exec_callback(job.update_callback)

            while self.jobs_finished_map:
                job_id, job = self.jobs_finished_map.popitem()
                logging.info(f'Job {job_id}: calling finish callback')
                exec_callback(job.finish_callback)

    # telegram related
    async def start_bot_session(self):
        logging.debug(f'start session...')
        self.telegram_session = get_client_session()
        res = await self.telegram_session.get(get_tg_endpoint('getMe'))
        logging.debug(f'session started...')
        await self.telegram_setMyCommands()
        asyncio.create_task(self.callback_loop())

    async def telegram_setMyCommands(self):
        for chat_id in allowed_chat_ids:
            logging.debug(f'set command for chat: {chat_id}')
            res = await self.telegram_session.post(
                get_tg_endpoint('setMyCommands'),
                json={
                    'commands': allowed_commands,
                    'scope': {"type": "chat", "chat_id": chat_id},
                },
            )
            logging.debug(f'set command done')

    async def telegram_sendMessage(self, msg):
        await self.telegram_session.post(get_tg_endpoint('sendMessage'), json=msg)


class WorkerStatus:
    WAITING = 'WAITING'
    RUNNING = 'RUNNING'
    OFFLINE = 'OFFLINE'


class Worker:
    def __init__(self, worker_id, status=None, timeout=10, post_destroy=None):
        self._status = None
        self.worker_id = worker_id
        self.timeout = int(timeout)
        self.post_destroy = post_destroy
        self.status = status
        asyncio.create_task(self.self_destroy())

    def refresh(self):
        logging.debug(f'Worker {self.worker_id}: refreshing...')
        self.timeout = 10

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        if value in (WorkerStatus.WAITING, WorkerStatus.RUNNING):
            old = self._status
            logging.debug(f'Worker {self.worker_id}: previous status: {old}')
            self._status = value
            self.refresh()
            if old == WorkerStatus.OFFLINE:
                asyncio.create_task(self.self_destroy())
                logging.info(f'Worker {self.worker_id}: back online')

        elif value == WorkerStatus.OFFLINE:
            self._status = value
            if self.post_destroy and callable(self.post_destroy):
                self.post_destroy()
            else:
                logging.debug(
                    f'Worker {self.worker_id}: Post destroyer not callable: {self.post_destroy}'
                )
            logging.info(f'Worker {self.worker_id}: {WorkerStatus.OFFLINE}')

        else:
            raise Exception(f'Worker {self.worker_id}: Unknown worker status: {value}')

    async def self_destroy(self):
        while self.timeout >= 0:
            await asyncio.sleep(3)
            self.timeout -= 3
            logging.debug(f'Worker {self.worker_id}: Counting down 3 for worker...')

        logging.info(f'Worker {self.worker_id}: taking offline now...')
        self.status = 'OFFLINE'


class JobStatus:
    WAITING = 'WAITING'
    RUNNING = 'RUNNING'
    FINISHED = 'FINISHED'
    FAILED = 'FAILED'


class JobSource:
    WEB = 'WEB'
    TG = 'TELEGRAM'


class Job:
    """
    job.job_source == 'WEB'/'TELEGRAM'
    job.job_details == {
        "job_id": <uuid>,
        "prompt": "",
        "width": 512,
        "height": 512
    }

    job.job_progress == {
        1: "localfilename1",
        2: "localfilename2",
        ...
    }
    job.job_result == {
        "result_img": "localfilename",
        "result_gif": "localfilename2"
    }
    """

    job_detail_fields = ('job_id', 'prompt', 'width', 'height', 'steps')

    def __init__(
        self,
        job_id: str = None,
        job_details: dict = None,
        assignee: Worker = None,
        job_source=JobSource.TG,
        tmp_dir: PurePath = None,
    ):
        self.job_id = job_id if job_id else str(uuid.uuid4())
        self.update_callback = None
        self.finish_callback = None
        self._job_source = ''
        self._job_details = {'job_id': self.job_id}
        self._job_status = JobStatus.WAITING

        self.job_assignee = assignee
        self.job_source = job_source

        if tmp_dir:
            tmp_dir = Path(tmp_dir)
            if not tmp_dir.exists():
                tmp_dir.mkdir(parents=True)
        self.job_tmpdir = Path(tmp_dir) if tmp_dir else None

        self.job_progress = {}
        self.job_result = {}
        if job_details:
            self.job_details = job_details

    def update_job_progress(self, index: int, filename: str):
        filepath = self.job_tmpdir / str(filename)
        if not filepath.exists() or not filepath.is_file():
            logging.error(
                f'Job {self.job_id}: progress file does not exist with index {index}: {filepath}, wont update progress'
            )
            raise AssertionError(
                f'Job {self.job_id}: progress file does not exist with index {index}: {filepath}, wont update progress'
            )

        self.job_progress[index] = filepath
        logging.debug(
            f'Job {self.job_id}: progress updated with index {index}, path: {filepath}'
        )

    @property
    def job_source(self):
        return self._job_source

    @job_source.setter
    def job_source(self, value):
        if value in (JobSource.TG, JobSource.WEB):
            self._job_source = value
            logging.info(f'Job {self.job_id}: from {value}')

        else:
            logging.error(f'Job {self.job_id}: unknown source: {value}')
            raise Exception(f'Job {self.job_id}: unknown source: {value}')

    @property
    def job_details(self):
        return self._job_details

    @job_details.setter
    def job_details(self, value: dict):
        if not value or not isinstance(value, dict):
            logging.error(f'Unknown job details: {value}')
            raise Exception(f'Unknown job details: {value}')

        else:
            logging.debug(f'Job {self.job_id}: old details: {self._job_details}')
            for key, item in value.items():
                if key in self.job_detail_fields:
                    self._job_details[key] = item

                else:
                    logging.info(
                        f'Job {self.job_id}: unknown field {key} for value {value} to update details'
                    )

            logging.debug(f'Job {self.job_id}: new details: {self._job_details}')

    @property
    def job_status(self):
        return self._job_status

    @job_status.setter
    def job_status(self, value):
        if self._job_status != value:
            logging.info(f'Updating job status from {self._job_status} to {value}')

        if value in (JobStatus.WAITING, JobStatus.RUNNING, JobStatus.FAILED):
            self._job_status = value

        elif value == JobStatus.FINISHED:
            self._job_status = value
            dump_json_to_file(self.job_details, self.job_tmpdir / 'desc.json')
            dump_json_to_file(self.job_progress, self.job_tmpdir / 'progress.json')
            dump_json_to_file(self.job_result, self.job_tmpdir / 'result.json')

        else:
            raise Exception(f'Job {self.job_id}: trying to set unknown status: {value}')


def get_path(v):
    return f'/bot{v}'


routes = web.RouteTableDef()
# Worker handlers
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

    > response: {}
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

            if job.job_status != JobStatus.RUNNING:
                job.job_status = JobStatus.RUNNING

        except StopIteration:
            raise AssertionError(
                f'Worker {worker.worker_id}: claims job {job_id} but job of this id is already lost.'
            )

    return web.Response()


@routes.get(get_path('/worker/{worker_id}/job'))
async def get_job(req: web.Request):
    """
    /worker/{worker_id}/task
    < response:
    job.job_details
    {
        "job_id": <uuid>,
        "prompt": "",
        "width": 512,
        "height": 512
    }
    or {}
    """
    # req_data = await req.json()
    worker_id = req.match_info['worker_id']
    app: BotServer = req.app
    worker = app.get_worker(worker_id)

    logging.info(f'Worker {worker.worker_id}: worker trying to get job')
    if worker.status != WorkerStatus.WAITING:
        logging.error(
            f'Worker {worker.worker_id}: worker is not WAITING but trying to get job'
        )
        return web.json_response({})

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
    res = web.json_response(job.job_details)
    await res.prepare(req)
    await res.write_eof()

    # confirm after sending successfully
    job.job_assignee = worker
    return res


@routes.post(get_path('/worker/{worker_id}/report'))
async def report(request: web.Request):
    """
    /worer/{worker_id}/report
    < request: form-multipart
    "job_id": <uuid>,
    "index": 123
    # finished or not
    "completed": false/true

    file: <file: 123.png>
    result_img: <file: abc.png>
    result_gif: <file: abc.mp4>

    """
    app: BotServer = request.app
    worker = app.get_worker(request.match_info['worker_id'])

    # get all fields
    progress_tmp = (
        progress_filename
    ) = job_id = index = result_img_tmp = result_gif_tmp = None
    completed = False

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

        elif field.name == 'result_gif':
            result_gif_tmp = app.tmp_dir / str(uuid.uuid4())
            await app.write_field(field, result_gif_tmp)

    # see if fields are complete, progress or result
    if not index or (
        None in (index, progress_tmp, progress_filename)
        and None in (result_img_tmp, result_gif_tmp)
    ):
        logging.error(
            f'Worker {worker.worker_id}: report is not complete, '
            f'job_id: {job_id}, index: {index}, progress_tmp: {progress_tmp}, '
            f'result_img_tmp: {result_img_tmp}, result_gif_tmp: {result_gif_tmp}, '
            f'skipping'
        )
        return web.Response(status=500, reason='progress content incomplete')
    job = app.get_job_by_id(job_id, raise_exception=True)

    # get progress filepath
    if not job.job_tmpdir:
        job.job_tmpdir = app.tmp_dir / str(job_id)
        if not job.job_tmpdir.exists():
            job.job_tmpdir.mkdir(parents=True)

    # write progress, update status if possible
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
        app.jobs_updated_map[job.job_id] = job
        return web.Response()

    elif completed:
        if result_img_tmp:
            ext = 'png'
            result_img_path = job.job_tmpdir / f'result.{ext}'
            result_img_tmp.rename(result_img_path)
            job.job_result['result_img'] = result_img_path.name

        if result_gif_tmp:
            ext = 'mp4'
            result_gif_path = job.job_tmpdir / f'result.{ext}'
            result_gif_tmp.rename(result_gif_path)
            job.job_result['result_gif'] = result_gif_path.name

        # write results
        job.job_status = JobStatus.FINISHED
        return web.Response()

    else:
        logging.error(
            f'Job {job_id}: Unknown report status, not progress nor completed, '
            f'completed: {completed}, index: {index}'
        )
        return web.Response(
            status=500, reason='Unknown report status, not progress nor completed'
        )


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


@routes.post(get_path(f'/{tg_bot_token}'))
async def telegram(req: web.Request):
    data = await req.json()
    print(json.dumps(data, ensure_ascii=False, indent=4))
    chat_id = get_param(data, ['message', 'chat', 'id'])
    msg_id = get_param(data, ['message', 'message_id'])
    if chat_id in allowed_chat_ids:
        app: BotServer = req.app
        msg = get_param(data, ['message'])
        try:
            await tg_prompt_handler(app, msg, chat_id)
        except Exception as e:
            res = web.Response()
            await res.prepare(req)
            await res.write_eof()
            error_msg = {'chat_id': chat_id, 'text': e.args[0] if e.args else 'Error'}
            if msg_id:
                error_msg.update(
                    {'reply_to_message_id': msg_id, 'allow_sending_without_reply': True}
                )
            await app.telegram_sendMessage(error_msg)
            return res

    return web.Response()


def tg_build_update_callback(
    app: BotServer, chat_id: int, reply_to_message_id: int, job_id
):
    # TODO: update/finish callback builder
    async def callback(msg=None):
        await app.telegram_sendMessage(
            {
                'chat_id': chat_id,
                'text': 'update' if not msg else msg,
                'reply_to_message_id': reply_to_message_id,
                'allow_sending_without_reply': True,
            }
        )
        logging.info(f'Job {job_id} update callback called')

    return callback


def tg_build_finish_callback(
    app: BotServer, chat_id: int, reply_to_message_id: int, job_id
):
    async def callback():
        logging.info(f'Job {job_id} calling finish callback')
        await app.telegram_sendMessage(
            {
                'chat_id': chat_id,
                'text': 'finish',
                'reply_to_message_id': reply_to_message_id,
                'allow_sending_without_reply': True,
            }
        )

    return callback


async def tg_prompt_handler(app: BotServer, msg: dict, chat_id):
    text = get_param(msg, 'text')
    logging.debug(f'Reading from message... {text}')
    words = re.split(r'\s+', text)

    if words[0].startswith('/prompt'):
        msg_id = msg.get('message_id')
        words = [word.replace('。', '.').replace('，', ',') for word in words[1:]]
        prompt = filter_prompt(' '.join(words))
        logging.info(f'Get prompt: {prompt}')
        job = Job(
            job_details={'prompt': prompt, 'width': 512, 'height': 512, 'steps': 50}
        )
        logging.info(f'Try adding new job {job.job_id}')
        app.add_job(
            job,
            update_callback=tg_build_update_callback(
                app, chat_id=chat_id, reply_to_message_id=msg_id, job_id=job.job_id
            ),
            finish_callback=tg_build_finish_callback(
                app, chat_id=chat_id, reply_to_message_id=msg_id, job_id=job.job_id
            ),
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
