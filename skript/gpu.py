#!/usr/bin/env python

from paderbox.utils.profiling import lprun

#@lprun
#def foo():

from plumbum import local, colors as c, SshMachine
import plumbum
import re
import os
import sys
# import numpy as np
import psutil
import tabulate
import json
from collections import defaultdict, OrderedDict
from multiprocessing import Pool
import time

# from joblib import Parallel, delayed

no_gpu_msg = 'localhost has no cmd nvidia-smi.'
no_host_msg = 'host does not exist.'

all_hosts = ['ntws{}'.format(i) for i in (5, 12, 25, 28, 29)]

class OrderedDefaultDict(OrderedDict):
    def __missing__(self, key):
        self.__dict__[key] = list()
        return self.__dict__[key]

def decode(a):
    r = re.compile(r'\|\s+\d\s+(\d{1,6})\s+\w\s+([\w/.-]+)\s+(\d+)([GMK]?iB)\s\|')

    #infos = defaultdict(list)
    infos = OrderedDefaultDict()
    for l in a.splitlines():
        if r.match(l):
            (pid, name, gpu_mem, gpu_mem_unit), = r.findall(l)
            # print(pid)

            infos['pid'] += [int(pid)]
            infos['process name'] += [name]
            infos['gpu_mem'] += [gpu_mem + ' ' + gpu_mem_unit]
            infos['user'] += [psutil.Process(int(pid)).username()]
            infos['cpu'] += [psutil.Process(int(pid)).cpu_percent(interval=0.0125)]
            infos['cpu_mem'] += [str(int(psutil.Process(int(pid)).memory_info().rss / 1024 / 1024)) + ' MiB']
            # print(yaml.dump(infos))
    return infos

class RefreshStdout:

    def __init__(self, use_curses=True):
        self.curses = curses

    def __enter__(self):
        if self.curses:
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            curses.start_color()
            curses.use_default_colors()
            self.output = ''
        return self

    def __exit__(self, *args):
        if self.curses:
            curses.echo()
            curses.nocbreak()
            curses.endwin()
        print(self.output)

    def refresh(self, tmp_out):
        if self.curses:
            self.stdscr.erase()
            self.stdscr.addstr(self.output)
            self.stdscr.addstr('\n\n')
            try:
                self.stdscr.addstr(tmp_out)
            except Exception:
                pass
            self.stdscr.refresh()
        else:
            print(tmp_out)
        time.sleep(2)

    def print(self, *args):
        if self.curses:
            self.output += ' '.join(args) + '\n'
        else:
            print(' '.join(args))


if len(sys.argv) >= 2 and sys.argv[1] != 'remote':
    import curses
    import yaml
    print('Remote execution: ', sys.argv)
    infos = OrderedDefaultDict()

    cmd = os.path.basename(sys.argv[0])

    hosts = sys.argv[1:]
    # hosts = [h for h in sys.argv[1:] if h in all_hosts]
    # hosts = all_hosts


    with RefreshStdout(True) as o:
        def foo(host):
            rem_std_out = no_host_msg
            try:
                with SshMachine(host) as rem:
                    rem_std_out = rem[cmd]['remote']()
            except plumbum.machines.session.SSHCommsError as e:
                print(e)
            return host, rem_std_out


        def foo_pseudo(host):
            a = local['nvidia-smi']()
            infos = decode(a)
            rem_std_out = yaml.dump(list(OrderedDict(infos).items()))
            return host, rem_std_out
        # foo = foo_pseudo
        # ThreadPool
        # from multiprocessing.pool import ThreadPool
        with Pool(processes=min([10, len(hosts)])) as pool:
            for host, rem_std_out in pool.imap_unordered(foo, hosts):
        #     for host in hosts:
        #         host, rem_std_out = foo(host)

                if no_gpu_msg in rem_std_out:
                    o.print(c.warn | no_gpu_msg.replace('localhost', host))
                    continue
                if no_host_msg in rem_std_out:
                    o.print(c.warn | no_host_msg.replace('host', host))
                    continue
                # print(rem_std_out)
                infos_host = yaml.load(rem_std_out)
                infos['host'] += [host]*len(infos_host[0][1])
                for key, value in infos_host:
                    infos[key] += value
                # except Exception:
                #     print(repr(rem_std_out))
                #     print(rem_std_out)
                #     print(infos_host)
                #     raise

                o.print(c.info | 'Found {}'.format(host))
            # print(c.info | 'Found {}'.format(host))

            o.refresh(tabulate.tabulate(infos,  infos.keys(), tablefmt='fancy_grid'))

    print(tabulate.tabulate(infos,  infos.keys(), tablefmt='fancy_grid'))
else:
    try:
        a = local['nvidia-smi']()
    except plumbum.commands.processes.CommandNotFound as e:
        print(no_gpu_msg)
        exit()

    if len(sys.argv) == 2 and sys.argv[1] == 'remote':
        import yaml
        infos = decode(a)
        # OrderedDefaultDict -> OrderedDict -> list -> yaml -> stdout
        print(yaml.dump(list(OrderedDict(infos).items())))
    else:

        print(tabulate.tabulate_formats)
        print(a)


        infos = decode(a)

        # infos = OrderedDict(sorted(infos.items()))
        print(tabulate.tabulate(infos,  infos.keys(), tablefmt='fancy_grid'))
        # print(yaml.dump(OrderedDict(infos)))
        #print(tabulate.tabulate(infos,  infos.keys(), tablefmt='orgtbl'))
        #print(tabulate.tabulate(infos,  infos.keys()))
    # foo()
