from asyncio import run
from logging import getLogger
from coder.utils import common

async def main():
    logger = getLogger(__name__)
    logger.info('Hello ...')

if __name__ == '__main__':
    run(main())