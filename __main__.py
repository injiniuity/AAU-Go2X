import asyncio
import sys

from .app import main


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nQuit")
        sys.exit(0)
