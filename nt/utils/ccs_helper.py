import sh
import pandas as pd
from io import StringIO
import re
import time
import numpy as np
from tqdm import tqdm


def ccsinfo(host=None, use_ssh=True):
    """ Wrapper for ccsinfo.

    :param host: PC2 ssh host alias.
    """
    common_args = ['-s', '--mine', '--fmt=%.R%.60N%.4w%P%D%v', '--raw']
    if use_ssh:
        host = 'pc2' if host is None else host
        ret = sh.ssh(host, 'ccsinfo', *common_args)
    else:
        ret = sh.ccsinfo(*common_args)
    return pd.read_csv(StringIO(ret.stdout.decode('utf-8')),
                       delim_whitespace=True,
                       names=['id', 'name', 'status', 'start_time',
                              'time_limit', 'runtime'])


def ccskill(request_id, host=None):
    """ Wrapper for ccskill.

    :param request_id: Single request id or list of request ids.
    :param host: PC2 ssh host alias.
    """
    host = 'pc2' if host is None else host
    if isinstance(request_id, (str, int)):
        request_id = [request_id]
    print(sh.ssh(host, 'ccskill', request_id))


def get_job_id(allocation_result):
    """ Extract cluster job ID from array job allocation string.

    Args:
        allocation_result: Result just as from ``job_result = sh.ssh(command)``.

    Returns:

    """
    try:
        regex_job_id = re.compile(r'PM:Array\s([0-9]*)')
        return re.search(
            regex_job_id,
            allocation_result.stdout.decode('utf-8')
        ).groups()[0]
    except AttributeError:
        regex_job_id = re.compile(r'Request\s\(([0-9]*)')
        return re.search(
            regex_job_id,
            allocation_result.stdout.decode('utf-8')
        ).groups()[0]


def _test_finished(job_ids, host, use_ssh):
    """ Expects list of job ids as strings. """
    job_ids = [str(_id) for _id in job_ids]
    df = ccsinfo(host=host, use_ssh=use_ssh)
    jobs = df[df['id'].apply(lambda x: str(x) in job_ids)]
    next_ids = []
    new_finished_jobs = 0
    for idx, job in jobs.iterrows():
        if not job['status'] == 'STOPPED':
            next_ids.append(job['name'])
        else:
            new_finished_jobs += 1
    return next_ids, new_finished_jobs


def idle_while_jobs_are_running(
        job_ids, sleep_time=300, host='pc2', use_ssh=True
):
    """ Expects list of job ids as strings. """
    total_jobs = len(job_ids)
    p = tqdm(total=total_jobs, desc='Cluster jobs')
    while len(job_ids):
        time.sleep(sleep_time)
        job_ids, new_finished_jobs = _test_finished(job_ids, host, use_ssh)
        if new_finished_jobs:
            p.update(new_finished_jobs)


def idle_while_array_jobs_are_running(
        job_ids, sleep_time=300, host='pc2', use_ssh=True
):
    """ Idle after job array allocation to wait for remaining jobs.

    Args:
        job_ids: List of job_ids as strings.
        sleep_time: Should be more than 300 seconds.
            Otherwise, PC2 administration gets angry.
        host: PC2 ssh host alias.

    """
    regex_completed = re.compile(r'Completed\s*subjobs\s*\:\s([0-9]*)')
    regex_running = re.compile(r'Running\s*subjobs\s*\:\s([0-9]*)')
    regex_planned = re.compile(r'Planned\s*subjobs\s*\:\s([0-9]*)')
    regex_waiting = re.compile(r'Waiting\s*subjobs\s*\:\s([0-9]*)')

    while len(job_ids):
        completed = 0
        running = 0
        planned = 0
        waiting = 0
        remaining_job_ids = []
        for job_id in job_ids:
            try:
                if use_ssh:
                    res = sh.ssh(host, 'ccsinfo', job_id)
                else:
                    res = sh.ccsinfo(job_id)
                res = res.stdout.decode('utf-8')
                jobs_running, jobs_planned, jobs_waiting = 0, 0, 0
                if re.search(regex_completed, res) is not None:
                    jobs_completed = int(
                        re.search(regex_completed, res).groups()[0])
                    completed += jobs_completed
                if re.search(regex_running, res) is not None:
                    jobs_running = int(
                        re.search(regex_running, res).groups()[0])
                    running += jobs_running
                if re.search(regex_planned, res) is not None:
                    jobs_planned = int(
                        re.search(regex_planned, res).groups()[0])
                    planned += jobs_planned
                if re.search(regex_waiting, res) is not None:
                    jobs_waiting = int(
                        re.search(regex_waiting, res).groups()[0])
                    waiting += jobs_waiting
                if np.sum([jobs_running, jobs_planned, jobs_waiting]) > 0:
                    remaining_job_ids.append(job_id)

            except Exception as e:
                message = 'Could not parse stats for job id {}: {}'
                print(message.format(job_id, e))

        print('Completed: {}\nRunning: {}\nPlanned: {}\nWaiting: {}'.format(
            completed, running, planned, waiting
        ))
        job_ids = remaining_job_ids
        time.sleep(sleep_time)
