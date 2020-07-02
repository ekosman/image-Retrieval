import logging
import sys


def register_logger():
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)

    logging.basicConfig(format="%(asctime)s %(message)s",
                        handlers=[
                            logging.StreamHandler(stream=sys.stdout)
                        ],
                        level=logging.INFO,
                        )
    logging.root.setLevel(logging.INFO)