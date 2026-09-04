from src.config import PROJECT_NAME
from src.utils.logger import get_logger


logger = get_logger(__name__)


def main() -> None:
    logger.info("%s AI subsystem started successfully", PROJECT_NAME)


if __name__ == "__main__":
    main()
