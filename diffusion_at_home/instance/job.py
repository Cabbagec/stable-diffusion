import json
import logging
import uuid
from pathlib import Path, PurePath
from typing import Union

from diffusion_at_home.utils import exec_callback


def dump_json_to_file(dict_json: dict, file):
    with open(file, 'w+') as f:
        json.dump(dict_json, ensure_ascii=False, indent=4, fp=f)


class JobStatus:
    WAITING = 'WAITING'
    RUNNING = 'RUNNING'
    FINISHED = 'FINISHED'
    FAILED = 'FAILED'
    ABORT = 'ABORTING'


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

    job_detail_fields = (
        'job_id',
        'prompt',
        'width',
        'height',
        'steps',
        'guidance_scale',
        'seed',
        # enable/disable animation
        'animation',
        # scale factor, 2/3/4
        'upscale',
    )

    def __init__(
        self,
        job_source,
        job_details: dict,
        job_resource_update_callback: callable,
        job_progress_callback: callable,
        job_finish_callback: callable,
        chat_id: Union[str, int] = None,
        chat_cmd_msg_id: str = None,
        job_id: str = None,
        # assignee: Worker = None,
        # tmp_dir: PurePath = None,
    ):
        """
        job_id:             if not provided, we'll generate a random one for ourselves
        job_details:        json dict, {'job_id': '...', ...}, see above
        job_assignee:       to which worker this job is assigned to
        job_source:         from where was this job created
        
        chat_id:            telegram source specific
        """

        self.job_id = job_id if job_id else str(uuid.uuid4())
        self._job_source = ''
        self._job_details = {'job_id': self.job_id}
        self._job_status = JobStatus.WAITING

        self.job_assignee = None
        self.job_source = job_source
        # chat_id for telegram source
        self.chat_id = chat_id
        self.chat_cmd_msg_id = chat_cmd_msg_id

        # self.animation_sent = False
        # self.upscaled_sent = False

        # if tmp_dir:
        #     tmp_dir = Path(tmp_dir)
        #     if not tmp_dir.exists():
        #         tmp_dir.mkdir(parents=True)
        # self.job_tmpdir = Path(tmp_dir) if tmp_dir else None
        self.job_tmpdir = None

        self.job_progress = {}
        self.job_result = {}
        if job_details:
            self.job_details = job_details

        # progress callbacks
        self.job_progress_callback = job_progress_callback
        self.job_finish_callback = job_finish_callback

        # update message and others
        self.update_message_id = ''
        # inline keyboard
        self.other_resources = [
            #
            'animation',
            # broken
            # 'Upscale x2',
            # 'Upscale x3',
            'Upscale x4',
        ]

        # job resources callbacks
        self._job_resources = {}
        self.job_resource_update_callback = job_resource_update_callback

    def get_inline_keyboard(self):
        inline_keyboard = {}
        for resource in self.other_resources:
            if 'animation' == resource:
                inline_keyboard[1] = [
                    {
                        'text': 'Generate process animation',
                        'callback_data': f'animation:{self.job_id}',
                    }
                ]

            if resource.lower().startswith('upscale'):
                if not inline_keyboard.get(2):
                    inline_keyboard[2] = []

                inline_keyboard[2].append(
                    {'text': resource, 'callback_data': f'{resource}:{self.job_id}'}
                )
        return list(inline_keyboard.values())

    def update_job_progress(self, index: int, filename: str):
        filepath = self.job_tmpdir / str(filename)
        if not filepath.exists() or not filepath.is_file():
            logging.error(
                f'Job {self.job_id}: progress file does not exist with index {index}: {filepath}, wont update progress'
            )
            raise AssertionError(
                f'Job {self.job_id}: progress file does not exist with index {index}: {filepath}, wont update progress'
            )

        self.job_progress[index] = filename
        logging.info(
            f'Job {self.job_id}: progress updated with index {index}, path: {filepath}'
        )
        logging.info(f'calling job progress callbacks')
        exec_callback(self.job_progress_callback, self)

    def update_job_result(self, tmp_file: PurePath):
        tmp_file = Path(tmp_file)
        result_img_path = self.job_tmpdir / f'result.png'
        tmp_file.rename(result_img_path)
        self.job_result['result_img'] = result_img_path.name
        self.job_status = JobStatus.FINISHED
        exec_callback(self.job_finish_callback, self)

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
        if (
            (
                self._job_status == JobStatus.WAITING
                and value
                in (
                    JobStatus.WAITING,
                    JobStatus.ABORT,
                    JobStatus.RUNNING,
                    JobStatus.FAILED,
                )
            )
            or (
                self._job_status == JobStatus.RUNNING
                and value
                in (
                    JobStatus.RUNNING,
                    JobStatus.ABORT,
                    JobStatus.FINISHED,
                    JobStatus.FAILED,
                )
            )
            or (
                self._job_status == JobStatus.FINISHED
                and value in (JobStatus.FINISHED,)
            )
            or (self._job_status == JobStatus.FAILED and value in (JobStatus.FAILED,))
            or (
                self._job_status == JobStatus.ABORT
                and value in (JobStatus.ABORT, JobStatus.FINISHED, JobStatus.FAILED)
            )
        ):
            # status check
            pass

        else:
            logging.error(
                f'Cannot update job status from {self._job_status} to {value}'
            )
            return

        if self._job_status != value:
            logging.info(f'Updating job status from {self._job_status} to {value}')
            self._job_status = value
        else:
            return

        if value in (JobStatus.WAITING, JobStatus.RUNNING, JobStatus.ABORT):
            pass

        elif value == JobStatus.FINISHED:
            self._job_status = value
            dump_json_to_file(self.job_details, self.job_tmpdir / 'desc.json')
            dump_json_to_file(self.job_progress, self.job_tmpdir / 'progress.json')
            dump_json_to_file(self.job_result, self.job_tmpdir / 'result.json')

        elif value == JobStatus.FAILED:
            self._job_status = value
            # TODO: do cleanups

        else:
            raise Exception(f'Job {self.job_id}: trying to set unknown status: {value}')

    def update_job_resource(self, resource_type, resource_path, filename):
        if resource_type not in self.other_resources:
            logging.error(
                f'Unknown resource type: {resource_type}, path: {resource_path}, won\'t update'
            )
            return

        resource_path = Path(resource_path)
        file_path = self.job_tmpdir / filename
        resource_path.rename(file_path)

        self._job_resources[resource_type] = file_path
        logging.info(
            f'Job {self.job_id} resource updated, type: {resource_type}, path: {resource_path}, executing callbacks'
        )
        exec_callback(self.job_resource_update_callback, self, resource_type, file_path)
