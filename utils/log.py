import os
import argparse
import logging
from logging import Filter
from logging.handlers import QueueHandler, QueueListener

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Queue


def setup_primary_logging(log_file_path: str) -> Queue:
    """
    Global logging is setup using this method. In a distributed setup, a multiprocessing queue is setup
    which can be used by the workers to write their log messages. This initializers respective handlers
    to pick messages from the queue and handle them to write to corresponding output buffers.
    Parameters
    ----------
    log_file_path : ``str``, required
        File path to write output log
    error_log_file_path: ``str``, required
        File path to write error log
    Returns
    -------
    log_queue : ``torch.multiprocessing.Queue``
        A log queue to which the log handler listens to. This is used by workers
        in a distributed setup to initialize worker specific log handlers(refer ``setup_worker_logging`` method).
        Messages posted in this queue by the workers are picked up and bubbled up to respective log handlers.
    """
    # Multiprocessing queue to which the workers should log their messages
    log_queue = Queue(-1)

    # Handlers for stream/file logging
    output_file_log_handler = logging.FileHandler(filename=str(log_file_path))

    formatter = logging.Formatter('%(asctime)s | %(message)s', '%Y-%m-%d %H:%M:%S')
    output_file_log_handler.setFormatter(formatter)

    # This listener listens to the `log_queue` and pushes the messages to the list of handlers specified.
    listener = QueueListener(log_queue, output_file_log_handler, respect_handler_level=True)
    listener.start()
    return log_queue


class WorkerLogFilter(Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f"Rank {self._rank} | {record.msg}"
        return True


def setup_worker_logging(rank: int, log_queue: Queue):
    """
    Method to initialize worker's logging in a distributed setup. The worker processes
    always write their logs to the `log_queue`. Messages in this queue in turn gets picked
    by parent's `QueueListener` and pushes them to respective file/stream log handlers.
    Parameters
    ----------
    rank : ``int``, required
        Rank of the worker
    log_queue: ``Queue``, required
        The common log queue to which the workers
    Returns
    -------
    features : ``np.ndarray``
        The corresponding log power spectrogram.
    """
    queue_handler = QueueHandler(log_queue)

    # Add a filter that modifies the message to put the rank in the log format
    worker_filter = WorkerLogFilter(rank)
    queue_handler.addFilter(worker_filter)

    root_logger = logging.getLogger()
    root_logger.addHandler(queue_handler)
    # Default logger level is WARNING, hence the change. Otherwise, any worker logs
    # are not going to get bubbled up to the parent's logger handlers from where the
    # actual logs are written to the output
    root_logger.setLevel(logging.WARNING)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def worker(rank: int, world_size: int, log_queue: Queue):
    setup_worker_logging(rank, log_queue)
    logging.debug("Test worker debug log")
    logging.info("Test worker info log")
    logging.warning("Test worker warning log")
    logging.error("Test worker error log")
    torch.cuda.set_device(rank)
    setup(rank, world_size)

    # Rest of the training code #

    #############################
    

if __name__ == "__main__":
    # Set multiprocessing type to spawn
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu-list", type=int, help="List of GPU IDs", nargs="+", required=True)

    args = parser.parse_args()
    world_size = len(args.gpu_list)
    log_queue = setup_primary_logging("out.log")
    mp.spawn(worker, args=(world_size, log_queue), nprocs=world_size)