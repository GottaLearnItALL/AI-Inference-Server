import asyncio, time


async def task(n):
    await asyncio.sleep(1)
    return n


async def main():
    start = time.perf_counter()
    results = await asyncio.gather(*[task(i) for i in range(5)])
    print(results, f'{time.perf_counter()-start:.2f}s')

asyncio.run(main())