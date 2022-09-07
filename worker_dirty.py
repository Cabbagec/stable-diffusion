import subprocess
from pathlib import Path

import httpx
import asyncio
import logging

from stable_run import load_model, get_opt, run as run_task, ProgressDisplayer
from functools import partial

token = 'helium'
url = f'https://bot.everdream.xyz/bot/worker/{token}'

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s]:%(levelname)s: %(message)s'
)


def get_endpoint(v):
    return f'{url}/{v}'


async def heartbeat(client, status_dict: dict):
    while True:
        logging.debug(f'heartbeating after 4...')
        try:
            await client.post(
                get_endpoint('heartbeat'),
                json={
                    'status': status_dict.get('status', 'WAITING'),
                    'job_id': status_dict.get('job_id'),
                },
            )
            await asyncio.sleep(5)
        except Exception:
            pass


async def get_job(client, job_dict: dict):
    while True:
        await asyncio.sleep(3)
        logging.info('Trying to get new job')
        res = await client.get(get_endpoint('job'))
        if not res:
            continue

        res_j = res.json()
        logging.info(f'job json received: {res_j}')
        if not res_j:
            continue

        job_id = res_j.get('job_id')
        if job_id not in job_dict:
            job_dict[job_id] = res_j
            break


async def send_progress(client, total_steps: int, updator: ProgressDisplayer):
    while True:
        await asyncio.sleep(3)
        try:
            job_id = updator.displayer_uuid
            if not updator.index_path:
                continue

            max_index, filepath = max(updator.index_path.items(), key=lambda kv: kv[0])
            if max_index == total_steps - 1:
                logging.info(f'generating animation for {job_id}...')
                if (
                    subprocess.run(
                        [
                            'ffmpeg',
                            '-framerate',
                            '24',
                            '-pattern_type',
                            'glob',
                            '-i',
                            '*.png',
                            '-c:v',
                            'libx264',
                            '-pix_fmt',
                            'yuv420p',
                            'output.mp4',
                        ],
                        cwd=filepath.parent,
                    ).returncode
                    == 0
                ):
                    animation_filepath = filepath.parent / "output.mp4"
                    logging.info(f'generated: {animation_filepath}')

                    with open(filepath, 'rb') as img_f, open(
                        animation_filepath, 'rb'
                    ) as gif_f:
                        await client.post(
                            get_endpoint('report'),
                            data={'completed': 'true', 'job_id': job_id},
                            files={
                                'result_img': ('result.png', img_f),
                                'result_gif': ('result.mp4', gif_f),
                            },
                        )
                else:
                    logging.info(f'generating animation failed... sending img only')
                    with open(filepath, 'rb') as img_f:
                        await client.post(
                            get_endpoint('report'),
                            data={'completed': 'true', 'job_id': job_id},
                            files={
                                'result_img': ('result.png', img_f),
                                # 'result_gif': ('result.mp4', gif_f),
                            },
                        )
                return

            with open(filepath, 'rb') as pf:
                await client.post(
                    get_endpoint('report'),
                    data={'index': max_index, 'job_id': job_id},
                    files={'file': pf},
                )
        except Exception as e:
            logging.error(f'send progress failed with {e}')
            logging.exception(e)


async def get_task_and_run(client, model, job_dict: dict, status_dict: dict):
    job_id, job_desc = job_dict.popitem()
    logging.info(f'Got new job {job_id}, {job_desc}, starting...')
    prompt = job_desc.get('prompt')
    width = int(job_desc.get('width', 0))
    height = int(job_desc.get('height', 0))
    steps = int(job_desc.get('steps', 0))

    opt = get_opt(prompt=prompt, width=width, height=height, steps=steps)
    updator = ProgressDisplayer(
        show_progress=False, save_progress=True, displayer_uuid=job_id
    )
    f = asyncio.get_event_loop().run_in_executor(
        None, partial(run_task, opt, model, updator)
    )
    status_dict.update({'status': 'RUNNING', 'job_id': job_id})
    await send_progress(client=client, total_steps=steps, updator=updator)


async def main():
    job_dict = {}
    model = load_model()
    status_dict = {}
    # model = object()
    client = httpx.AsyncClient(
        http2=True,
        proxies='http://127.0.0.1:8123',
        # proxies='http://127.0.0.1:8889',
        # verify='/etc/ssl/cert.pem'
    )
    asyncio.create_task(heartbeat(client=client, status_dict=status_dict))

    while True:
        await get_job(client, job_dict)
        # asyncio.create_task(get_task_and_run(model, job_dict))
        try:
            await asyncio.wait_for(
                get_task_and_run(client, model, job_dict, status_dict), timeout=120
            )
        except asyncio.TimeoutError:
            logging.error(f'current job timed out')

        finally:
            status_dict.update({'status': 'WAITING', 'job_id': None})

        await asyncio.sleep(1)


if __name__ == '__main__':
    asyncio.run(main())
