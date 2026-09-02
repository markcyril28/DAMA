"""Bounded native metadata batches for slow mounted filesystems.

The corpus admission transaction must stat dozens of independent replay
shards several times.  Python's ThreadPoolExecutor overlaps drvfs latency, but
constructing Futures and Python worker threads dominates the warmed decision.
This helper performs the same following ``stat(2)`` calls on short-lived C11
threads and joins every thread before returning, so the trainer never forks a
self-play pool while metadata helper threads are still alive.
"""

from cpython.bytes cimport PyBytes_AS_STRING
from libc.errno cimport errno
from libc.stdlib cimport free, malloc
from libc.threads cimport (
    thrd_create,
    thrd_join,
    thrd_start_t,
    thrd_success,
    thrd_t,
)
from posix.stat cimport stat as c_stat, struct_stat

import os


cdef struct _StatTask:
    const char **paths
    struct_stat *results
    int *errors
    Py_ssize_t start
    Py_ssize_t stop


cdef int _stat_range(void *raw_task) noexcept nogil:
    cdef _StatTask *task = <_StatTask *>raw_task
    cdef Py_ssize_t index
    for index in range(task.start, task.stop):
        task.errors[index] = 0
        if c_stat(task.paths[index], &task.results[index]) != 0:
            task.errors[index] = errno
    return 0


def stat_paths(paths, int workers=8):
    """Return exact following-stat fields and errno for each path, in order.

    Each success is ``((mode, dev, ino, size, mtime_ns), 0)``.  Each failure
    is ``(None, errno)``.  Python owns path encoding and exception construction;
    only independent system calls execute without the GIL.
    """

    cdef Py_ssize_t count = len(paths)
    cdef Py_ssize_t index
    cdef int worker_count
    cdef int worker
    cdef int join_result
    cdef list encoded
    cdef list output
    cdef const char **raw_paths = NULL
    cdef struct_stat *results = NULL
    cdef int *errors = NULL
    cdef _StatTask *tasks = NULL
    cdef thrd_t *threads = NULL
    cdef int *created = NULL
    cdef long long mtime_ns

    if count == 0:
        return []
    if workers < 1:
        raise ValueError("workers must be positive")
    worker_count = min(workers, count)
    encoded = [os.fsencode(path) for path in paths]

    raw_paths = <const char **>malloc(count * sizeof(const char *))
    results = <struct_stat *>malloc(count * sizeof(struct_stat))
    errors = <int *>malloc(count * sizeof(int))
    tasks = <_StatTask *>malloc(worker_count * sizeof(_StatTask))
    if worker_count > 1:
        threads = <thrd_t *>malloc((worker_count - 1) * sizeof(thrd_t))
        created = <int *>malloc((worker_count - 1) * sizeof(int))
    if (
        raw_paths == NULL
        or results == NULL
        or errors == NULL
        or tasks == NULL
        or (worker_count > 1 and (threads == NULL or created == NULL))
    ):
        free(raw_paths)
        free(results)
        free(errors)
        free(tasks)
        free(threads)
        free(created)
        raise MemoryError()

    try:
        for index in range(count):
            raw_paths[index] = PyBytes_AS_STRING(encoded[index])
        for worker in range(worker_count):
            tasks[worker].paths = raw_paths
            tasks[worker].results = results
            tasks[worker].errors = errors
            tasks[worker].start = count * worker // worker_count
            tasks[worker].stop = count * (worker + 1) // worker_count
        for worker in range(worker_count - 1):
            created[worker] = 0

        # Use the calling thread for one partition and create at most 31 helpers
        # for the production width of 32. A failed thread creation simply runs
        # that partition synchronously. Every successful helper is joined before
        # Python can start the next self-play fork.
        with nogil:
            for worker in range(worker_count - 1):
                if thrd_create(
                    &threads[worker],
                    <thrd_start_t>_stat_range,
                    <void *>&tasks[worker],
                ) == thrd_success:
                    created[worker] = 1
                else:
                    _stat_range(<void *>&tasks[worker])
            _stat_range(<void *>&tasks[worker_count - 1])
            for worker in range(worker_count - 1):
                if created[worker]:
                    join_result = thrd_join(threads[worker], NULL)
                    if join_result != thrd_success:
                        # The result is intentionally ignored here.  A valid
                        # thread handle is joinable exactly once; this branch
                        # exists only to make the operation explicit to Cython.
                        pass

        output = []
        for index in range(count):
            if errors[index]:
                output.append((None, errors[index]))
                continue
            mtime_ns = (
                <long long>results[index].st_mtim.tv_sec * 1000000000
                + <long long>results[index].st_mtim.tv_nsec
            )
            output.append(((
                int(results[index].st_mode),
                int(results[index].st_dev),
                int(results[index].st_ino),
                int(results[index].st_size),
                int(mtime_ns),
            ), 0))
        return output
    finally:
        free(raw_paths)
        free(results)
        free(errors)
        free(tasks)
        free(threads)
        free(created)
