import logging
from collections import deque
from pathlib import Path, PurePath
from typing import Dict, Deque

import httpx
from aiohttp import web, BodyPartReader

from diffusion_at_home.config import cache_dir, allowed_commands, allowed_chat_ids
from diffusion_at_home.instance.job import Job, JobStatus
from diffusion_at_home.instance.worker import Worker, WorkerStatus
from diffusion_at_home.utils import get_tg_endpoint


class ServerException(Exception):
    pass


class ServerJobWaitingException(ServerException):
    pass


class BotServer(web.Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # worker_id: Worker instance
        self.workers: Dict[str, Worker] = dict()
        self.jobs_queue: Deque[Job] = deque()

        _cache_dir = Path(cache_dir)
        if _cache_dir.exists():
            if _cache_dir.is_file():
                raise Exception(f'Path {cache_dir} exists and is not a directory')
        else:
            _cache_dir.mkdir(parents=True)
        self.tmp_dir = _cache_dir

        # TODO: rate limit from telegram
        self.rate_limits = {'TELEGRAM': 0}

        self.telegram_session = httpx.AsyncClient(http2=False)
        self.set_ssl_cert = None

    def add_job(self, job: 'Job'):
        workers = self.get_workers_by_status(WorkerStatus.WAITING)
        if not workers:
            available_workers = len(self.get_workers_by_status(WorkerStatus.RUNNING))
            if available_workers:
                logging.info(f'Waiting, Num of available workers: {available_workers}')
                self.jobs_queue.append(job)
                q_len = len(
                    list(
                        e for e in self.jobs_queue if e.job_status == JobStatus.WAITING
                    )
                )
                raise ServerJobWaitingException(
                    f'🤩 Job added, waiting. Current available workers: {available_workers}, queue length: {q_len}',
                    q_len,
                )

            else:
                logging.info(f'No workers available for job {job.job_id}, abandon')
                job.job_status = JobStatus.FAILED
                raise ServerException(
                    f'Sorry 😢, no workers are available currently. '
                    f'Please try again after workers are available.'
                )
        self.jobs_queue.append(job)
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
            logging.error(f'Cannot find job by id: {job_id}')
            if raise_exception:
                raise Exception(f'Cannot find job by id: {job_id}')
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

    # telegram related
    async def start_bot_session(self):
        logging.debug(f'start session...')
        res = await self.telegram_session.get(get_tg_endpoint('getMe'))
        logging.debug(f'session started...')
        await self.telegram_setMyCommands()

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

    async def telegram_sendMessage(self, msg: dict):
        await self.telegram_session.post(get_tg_endpoint('sendMessage'), json=msg)
