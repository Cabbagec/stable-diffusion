import httpx
import asyncio
import logging

token = 'tokenexample'
url = f'https://example.com/worker/{token}'

logging.basicConfig(
    level=logging.DEBUG, format='[%(asctime)s]:%(levelname)s: %(message)s'
)


async def send(client, url, msg=''):
    if isinstance(msg, dict):
        return await client.post(url, json=msg)
    elif isinstance(msg, str):
        return await client.post(url, content=msg)
    else:
        return await client.post(url)


async def heartbeat(client, task_queue):
    while True:
        print(f'heartbeating after 5...')
        await asyncio.sleep(5)
        content = await send(client, url + '/heartbeat')
        if content:
            await task_queue.put(content)

        print('heartbeat done...')


async def send_results(client, results_queue):
    while True:
        content = await results_queue.get()
        print(f'results: {content} received')
        await send(client, url + '/results', content)


async def get_task_and_run(task_queue):
    while True:
        task = await task_queue.get()
        print(f'task: {task} done')
        pass


async def run(task_queue):
    i = 1
    while True:
        print(f'sleep 3 then put {i} in task queue')
        await asyncio.sleep(3)
        await task_queue.put(i)


async def main():
    results_queue = asyncio.Queue()
    task_queue = asyncio.Queue()
    client = httpx.AsyncClient(http2=True)
    await asyncio.gather(
        heartbeat(client, task_queue),
        send_results(client, results_queue),
        get_task_and_run(task_queue),
        mock(task_queue),
    )


if __name__ == '__main__':
    asyncio.run(main())
