import asyncio
import logging
from collections import defaultdict


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
        # job_id: ['animation'/'upscalex2'/'upscalex3'/upscalex4']
        self.resources_to_fetch = defaultdict(list)
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
