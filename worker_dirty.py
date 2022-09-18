import asyncio
import json
import logging
import uuid
from functools import partial
from pathlib import Path

import httpx

from stable_run import (
    load_model,
    get_opt,
    run as run_task,
    ProgressDisplayer,
    generate_animation,
    generate_upscaled,
)

token = 'helium'
url = f'https://bot.everdream.xyz/bot/worker/{token}'
realesrgan_dir = '/Repositories/realesrgan'
tmp_save_path = '/tmp'

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s]:%(levelname)s: %(message)s'
)


def get_endpoint(v):
    return f'{url}/{v}'


async def heartbeat(client, status_dict: dict):
    while True:
        logging.info(f'heartbeating after 5...')
        try:
            await asyncio.sleep(5)
            await client.post(
                get_endpoint('heartbeat'),
                json={
                    'status': status_dict.get('status', 'WAITING'),
                    'job_id': status_dict.get('job_id'),
                },
            )
            # abort job
            # if r.json().get('abort'):
            #     status_dict['abort'] = True

        except Exception as e:
            logging.exception(e)
            pass


async def get_job(client, job_dict: dict):
    while True:
        await asyncio.sleep(3)
        logging.debug('Trying to get new job')
        try:
            res = await client.get(get_endpoint('job'))
            if not res:
                continue

            res_j = res.json()
            logging.debug(f'job json received: {res_j}')
            if not res_j:
                continue

            job_id = res_j.get('job_id')
            if job_id not in job_dict:
                job_dict[job_id] = res_j
                break
        except Exception as e:
            logging.error(f'failed to get job')
            logging.exception(e)


async def send_progress(
    client, total_steps: int, updator: ProgressDisplayer, params, status_dict
):
    while True:
        await asyncio.sleep(3)
        try:
            job_id = updator.displayer_uuid
            if not updator.index_path:
                # job hasn't started
                continue

            # abort on abort flag appearing
            if status_dict.get('abort'):
                status_dict.pop('abort', None)
                updator.abort_on_next()
                asyncio.create_task(
                    client.post(
                        get_endpoint('error'),
                        json={'job_id': job_id, 'error': 'Job aborted'},
                    )
                )
                break

            # get total steps of job
            total_steps = updator.total_steps
            if total_steps:
                params['steps'] = total_steps

            max_index, filepath = max(updator.index_path.items(), key=lambda kv: kv[0])
            if max_index >= total_steps - 1:
                logging.info(f'generation of last pic done')
                with open(filepath, 'rb') as img_f:
                    r = await client.post(
                        get_endpoint('report'),
                        data={
                            'completed': 'true',
                            'job_id': job_id,
                            'job_details': json.dumps(params),
                        },
                        files={'result_img': ('result.png', img_f)},
                    )
                    break
            else:
                logging.info(f'sending step: {max_index}')
                with open(filepath, 'rb') as pf:
                    r = await client.post(
                        get_endpoint('report'),
                        data={
                            'index': max_index,
                            'job_id': job_id,
                            'job_details': json.dumps(params),
                        },
                        files={'file': pf},
                    )

            if r.json().get('abort'):
                status_dict['abort'] = True

        except Exception as e:
            logging.error(f'send progress failed with {e}')
            logging.exception(e)


def get_resources(job_desc):
    resources = job_desc.get('resources')
    job_id = job_desc.get('job_id')
    if not resources or not job_id:
        return

    search_path = Path(tmp_save_path) / job_id
    job_result = search_path / 'result.json'
    if not (search_path.exists() and search_path.is_dir() and job_result.exists()):
        logging.error(
            f'Cannot find tmp result for job {job_id}, search path: {job_result}'
        )
        return

    with open(job_result, 'r') as f:
        result = json.load(fp=f)
        logging.info(f'job result detail at {job_result} loaded: {result}')

    if not result:
        logging.error(f'Tmp result for job {job_id} empty')
        return

    last_img = result.get('last_img')
    last_index = result.get('last_index')
    if None in (last_index, last_img):
        logging.error(
            f'Cannot find last image {last_img} or last {last_index} index for job {job_id}'
        )
        return

    results = {}
    last_img = Path(last_img)
    for resource in resources:
        if 'animation' == resource:
            results[resource] = generate_animation(last_img.parent)

        elif 'Upscale x2' == resource:
            results[resource] = generate_upscaled(last_img, factor=2)

        elif 'Upscale x3' == resource:
            results[resource] = generate_upscaled(last_img, factor=3)

        elif 'Upscale x4' == resource:
            results[resource] = generate_upscaled(last_img, factor=4)

        else:
            logging.error(f'Unknown resource: {resource}, skipping...')

    return results


async def send_resource(client, job_id, resource_name, resource_path):
    resource_path = Path(resource_path)
    with open(resource_path, 'rb') as f:
        await client.post(
            get_endpoint('resource'),
            data={'job_id': job_id, 'resource_type': resource_name},
            files={'resource': (resource_path.name, f)},
        )
        logging.info(f'resource {resource_path} sent')


async def get_task_and_run(client, model, job_dict: dict, status_dict: dict):
    job_id, job_desc = job_dict.popitem()
    if job_desc.get('resources'):
        # dict of resource paths
        # just send resources and exit current job
        for resource_type, resource_path in get_resources(job_desc).items():
            logging.info(
                f'sending resource {resource_type} at {resource_path} to server on job {job_id}'
            )
            await send_resource(client, job_id, resource_type, resource_path)

        return

    logging.info(f'Got new job {job_id}, {job_desc}, starting...')
    status_dict.update({'job_id': job_id})
    prompt = job_desc.get('prompt')
    width = int(job_desc.get('width', 0))
    height = int(job_desc.get('height', 0))
    steps = min(100, max(0, int(job_desc.get('steps', 0))))
    guidance_scale = min(8.0, max(1.0, float(job_desc.get('guidance_scale', 4))))
    seed = min(
        4294967295,
        max(0, int(job_desc.get('seed', uuid.uuid4().int & ((1 << 32) - 1)))),
    )
    if width % 64 != 0 or height % 64 != 0:
        raise Exception(f'Width and height must be factors of 64.')

    if (width * height) > (512 * 512 * 1.25):
        raise Exception(f'Image too large, try use smaller width and height')

    params = {
        'prompt': prompt,
        'width': width,
        'height': height,
        'steps': steps,
        'scale': guidance_scale,
        'seed': seed,
    }
    opt = get_opt(**params)
    updator = ProgressDisplayer(
        show_progress=False,
        save_progress=True,
        displayer_uuid=job_id,
        save_path=tmp_save_path,
    )
    f = asyncio.get_event_loop().run_in_executor(
        None, partial(run_task, opt, model, updator)
    )
    status_dict.update({'status': 'RUNNING', 'job_id': job_id})
    await send_progress(
        client=client,
        total_steps=steps,
        updator=updator,
        params=params,
        status_dict=status_dict,
    )


async def main():
    job_dict = {}
    model = load_model()
    status_dict = {}
    # model = object()
    client = httpx.AsyncClient(
        http2=False,
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
                get_task_and_run(client, model, job_dict, status_dict), timeout=200
            )
        except asyncio.TimeoutError:
            logging.error(f'current job timed out')

        except Exception as e:
            logging.error(e)
            asyncio.create_task(
                client.post(
                    get_endpoint('error'),
                    json={'job_id': status_dict.get('job_id'), 'error': str(e)},
                )
            )

        finally:
            status_dict.update({'status': 'WAITING', 'job_id': None})

        await asyncio.sleep(1)


if __name__ == '__main__':
    asyncio.run(main())
