"""Entry point from my app."""

from criteologging import criteo_logger


logger = criteo_logger.get_logger(__name__)


def greet(name="Anonymous"):
    """Greet people.

    Args:
      name: string, a name
    """
    return "Hello %s" % name


def main():
    """Main function."""
    logger.info(greet())


if __name__ == '__main__':
    main()
