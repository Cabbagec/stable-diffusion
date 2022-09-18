import json
import logging
import re
from pathlib import Path

from aiohttp import web

from diffusion_at_home.config import tg_bot_token, allowed_chat_ids
from diffusion_at_home.instance.app_server import (
    BotServer,
    ServerException,
    ServerJobWaitingException,
)
from diffusion_at_home.instance.job import Job, JobStatus, JobSource
from diffusion_at_home.instance.worker import Worker
from diffusion_at_home.utils import (
    get_tg_endpoint,
    get_param,
    filter_prompt,
    exec_callback,
)

routes = web.RouteTableDef()


def get_path(v):
    return f'/bot{v}'


@routes.post(get_path(f'/{tg_bot_token}'))
async def telegram(req: web.Request):
    data = await req.json()
    logging.info(f'msg: {data}')
    app: BotServer = req.app

    await tg_prompt_handler(app, data)
    await tg_callback_query(app, data)

    return web.Response()


def tg_build_update_callback(app: BotServer, chat_id: int, reply_to_message_id: int):
    async def callback(job: Job):

        # check if there's any actual progress
        if not job.job_progress:
            logging.error(
                f'Job {job.job_id} update callback called, but no progress found.'
            )
            return

        # find out the newest update
        max_index, max_progress_filename = max(
            job.job_progress.items(), key=lambda kv: kv[0]
        )
        details = job.job_details
        max_progress_filepath = job.job_tmpdir / max_progress_filename
        total_steps = details.get('steps')
        logging.info(f'Job {job.job_id} updating {max_index}/{total_steps}...')

        width, height = details.get('width'), details.get('height')
        steps, seed = details.get('steps'), details.get('seed')
        guidance_scale = details.get('guidance_scale') or details.get('scale')

        # create the progress message
        if not job.update_message_id:
            with open(max_progress_filepath, 'rb') as progress_f:
                result_res = await app.telegram_session.post(
                    get_tg_endpoint('sendPhoto'),
                    data={
                        'chat_id': str(chat_id),
                        'caption': f'{max_index}/{total_steps}',
                        'reply_to_message_id': str(reply_to_message_id),
                        'allow_sending_without_reply': 'true',
                        'reply_markup': json.dumps(
                            {
                                'inline_keyboard': [
                                    [
                                        {
                                            'text': 'Abort',
                                            'callback_data': f'abort:{job.job_id}',
                                        }
                                    ]
                                ]
                            }
                        ),
                    },
                    files={'photo': progress_f},
                )
                job.update_message_id = get_param(
                    result_res.json(), ['result', 'message_id']
                )
                return

        # update the existing progress message
        with open(max_progress_filepath, 'rb') as progress_f:
            await app.telegram_session.post(
                get_tg_endpoint('editMessageMedia'),
                data={
                    'chat_id': str(chat_id),
                    'message_id': str(job.update_message_id),
                    'media': json.dumps(
                        {
                            'type': 'photo',
                            'media': f'attach://{Path(max_progress_filepath).name}',
                            'caption': f'{max_index}/{total_steps}\n\n'
                            f'Seed: {seed}\n'
                            f'WxH: {width}x{height}\n'
                            f'Guidance scale: {guidance_scale}\n'
                            f'Steps: {steps}',
                        }
                    ),
                    'reply_markup': json.dumps(
                        {
                            'inline_keyboard': [
                                [
                                    {
                                        'text': 'Abort',
                                        'callback_data': f'abort:{job.job_id}',
                                    }
                                ]
                            ]
                        }
                    ),
                },
                files={Path(max_progress_filepath).name: progress_f},
            ),
        logging.info(f'Job {job.job_id} update callback done')

    return callback


def tg_build_finish_callback(app: BotServer, chat_id: int, reply_to_message_id: int):
    async def callback(job: Job):
        logging.info(f'Job {job.job_id} calling finish callback')
        details = job.job_details

        # find out finished result img and gif
        result_img = job.job_tmpdir / job.job_result.get('result_img')
        # result_gif = job.job_tmpdir / job.job_result.get('result_gif')
        job_prompt = details.get('prompt', '')
        width, height = details.get('width'), details.get('height')
        steps, seed = details.get('steps'), details.get('seed')
        guidance_scale = details.get('guidance_scale') or details.get('scale')
        if not result_img.exists():
            logging.error(f'Job {job.job_id} finished, but missing file: {result_img}')
            raise Exception(f'Error: Result file missing')

        with open(result_img, 'rb') as rimg_f:
            if job.update_message_id:
                await app.telegram_session.post(
                    get_tg_endpoint('editMessageMedia'),
                    data={
                        'chat_id': str(chat_id),
                        'message_id': job.update_message_id,
                        'media': json.dumps(
                            {
                                'type': 'photo',
                                'media': f'attach://{result_img.name}',
                                'caption': f'{job_prompt}\n\n'
                                f'Seed: {seed}\n'
                                f'WxH: {width}x{height}\n'
                                f'Guidance scale: {guidance_scale}\n'
                                f'Steps: {steps}',
                            }
                        ),
                        'reply_markup': json.dumps(
                            {'inline_keyboard': job.get_inline_keyboard()}
                        ),
                    },
                    files={result_img.name: rimg_f},
                )
            # send result img
            else:
                await app.telegram_session.post(
                    get_tg_endpoint('sendPhoto'),
                    data={
                        'chat_id': str(chat_id),
                        'caption': f'{job_prompt}\n\n'
                        f'Seed: {seed}\n'
                        f'WxH: {width}x{height}\n'
                        f'Guidance scale: {guidance_scale}\n'
                        f'Steps: {steps}',
                        'reply_to_message_id': str(reply_to_message_id),
                        'allow_sending_without_reply': 'true',
                        'reply_markup': json.dumps(
                            {'inline_keyboard': job.get_inline_keyboard()}
                        ),
                    },
                    files={'photo': rimg_f},
                )
        logging.debug(f'Job {job.job_id} finish callback done')

    return callback


def build_resource_update_callback(app: BotServer):
    async def tg_job_resource_update_callback(job: Job, resource_type, resource_path):
        logging.info(f'Calling telegram callback for job resource update...')
        resource_path = Path(resource_path)
        if not resource_path.exists() or not resource_path.is_file():
            logging.error(f'unknown resource type')
            await app.telegram_sendMessage(
                {
                    'chat_id': job.chat_id,
                    'reply_to_message_id': job.update_message_id,
                    'text': f'Failed to get resource {resource_type}',
                    'allow_sending_without_reply': True,
                }
            )
            return

        # update keyboard
        try:
            job.other_resources.remove(resource_type)
        except ValueError:
            logging.error(f'Resource {resource_type} may have already been updated')

        exec_callback(
            app.telegram_session.post,
            get_tg_endpoint('editMessageReplyMarkup'),
            data={
                'chat_id': job.chat_id,
                'message_id': job.update_message_id,
                'reply_markup': json.dumps(
                    {'inline_keyboard': job.get_inline_keyboard()}
                ),
            },
        )

        with open(resource_path, 'rb') as f:
            if resource_type.lower().startswith('upscale'):
                logging.info(f'sending upscale resource: {resource_type}')
                await app.telegram_session.post(
                    get_tg_endpoint('sendDocument'),
                    data={
                        'chat_id': str(job.chat_id),
                        'reply_to_message_id': str(job.update_message_id),
                        'allow_sending_without_reply': 'true',
                    },
                    files={'document': (resource_path.name, f)},
                )
                logging.info(f'sending upscale resource done: {resource_type}')

            elif resource_type.lower() == 'animation':
                logging.info(f'sending animation resource: {resource_type}')
                details = job.job_details
                await app.telegram_session.post(
                    get_tg_endpoint('sendAnimation'),
                    data={
                        'chat_id': job.chat_id,
                        'caption': f'{details.get("prompt")}\n\n'
                        f'Seed: {details.get("seed")}\n'
                        f'WxH: {details.get("width")}x{details.get("height")}\n'
                        f'Guidance Scale: {details.get("guidance_scale") or details.get("scale")}\n'
                        f'Steps: {details.get("steps")}',
                    },
                    files={'animation': f},
                )

            else:
                logging.error(
                    f'Unknown resource type {resource_type}, file: {resource_path}, won\'t send to chat'
                )

    return tg_job_resource_update_callback


def tg_build_failed_callback(app: BotServer):
    async def callback(job: Job):
        r = await app.telegram_session.post(
            get_tg_endpoint('editMessageReplyMarkup'),
            data={
                'chat_id': job.chat_id,
                'message_id': job.update_message_id,
                'reply_markup': json.dumps({'inline_keyboard': []}),
            },
        )
        logging.error(f'========= try removing inline keyboard, result: {r.json()}')

    return callback


def build_other_message_callback(app: BotServer):
    async def callback(job: Job, message: str):
        await app.telegram_sendMessage(
            msg={
                'chat_id': job.chat_id,
                'text': message,
                'reply_to_message_id': job.update_message_id,
                'allow_sending_without_reply': True,
            }
        )

    return callback


async def tg_prompt_handler(app: BotServer, update: dict):
    chat_id = get_param(update, ['message', 'chat', 'id'])
    msg_id = get_param(update, ['message', 'message_id'])
    msg = get_param(update, ['message'])
    text = get_param(msg, 'text')

    if chat_id not in allowed_chat_ids:
        return

    try:
        logging.debug(f'Reading from message... {text}')
        words = re.split(r'\s+', text)

        if words[0].startswith('/prompt'):
            msg_id = msg.get('message_id')
            words = [word.replace('。', '.').replace('，', ',') for word in words[1:]]
            extra_param = {}
            prompt_words = []
            for word in words:
                if ':' in word:
                    extra_param.update(
                        dict(re.findall(r'([0-9a-zA-Z\-_]+):(\d+)', word))
                    )

                else:
                    prompt_words.append(word)

            prompt = filter_prompt(' '.join(prompt_words))
            logging.info(f'Get prompt: {prompt}')
            job = Job(
                job_source=JobSource.TG,
                chat_id=chat_id,
                chat_cmd_msg_id=msg_id,
                job_details={
                    'prompt': prompt,
                    'width': 512,
                    'height': 512,
                    'steps': 50,
                    **extra_param,
                },
                job_resource_update_callback=build_resource_update_callback(app),
                job_progress_callback=tg_build_update_callback(app, chat_id, msg_id),
                job_finish_callback=tg_build_finish_callback(app, chat_id, msg_id),
                job_failed_callback=tg_build_failed_callback(app),
                misc_message_callback=build_other_message_callback(app),
            )
            logging.info(f'Try adding new job {job.job_id} from chat {chat_id}')
            app.add_job(job)
            await app.telegram_sendMessage(
                {
                    'chat_id': chat_id,
                    'text': '🤩 Job added, waiting for generating...',
                    'reply_to_message_id': msg_id,
                    'allow_sending_without_reply': True,
                }
            )

    except ServerJobWaitingException as e:
        await app.telegram_sendMessage(
            {
                'chat_id': chat_id,
                'text': e.args[0] if e.args else 'Job added, waiting',
                'reply_to_message_id': msg_id,
                'allow_sending_without_reply': True,
            }
        )

    except ServerException as e:
        await app.telegram_sendMessage(
            {
                'chat_id': chat_id,
                'text': e.args[0] if e.args else f'Bot server unknown error: {e}',
                'reply_to_message_id': msg_id,
                'allow_sending_without_reply': True,
            }
        )

    except Exception as e:
        error_msg = {'chat_id': chat_id, 'text': e.args[0] if e.args else 'Error'}
        if msg_id:
            error_msg.update(
                {'reply_to_message_id': msg_id, 'allow_sending_without_reply': True}
            )
        await app.telegram_sendMessage(error_msg)


# TODO: Add callback query
async def tg_callback_query(app: BotServer, update: dict):
    chat_id = get_param(update, ['callback_query', 'message', 'chat', 'id'])
    message_id = get_param(update, ['callback_query', 'message', 'message_id'])
    if chat_id not in allowed_chat_ids:
        return

    callback_data = get_param(update, ['callback_query', 'data'])
    if not callback_data:
        return

    action, job_id, *_ = str(callback_data).split(':')
    logging.info(f'callback received, action: {action}, job: {job_id}')

    job = app.get_job_by_id(job_id)
    if action in ('animation', 'Upscale x2', 'Upscale x3', 'Upscale x4'):
        if not job:
            logging.error(f'cannot find job {job_id} for action {action}')
            return

        if not job.job_assignee:
            logging.error(f'cannot find job assignee')

        assignee: Worker = job.job_assignee
        assignee.resources_to_fetch[job_id].append(action)

        return

    if 'abort' == action:
        job.job_status = JobStatus.ABORT
        logging.warning(f'Trying to abort job...')
        return

    return
