#!/usr/bin/env python

from plumbum import local
import re, os, numpy as np
import psutil, yaml, tabulate
from collections import defaultdict, OrderedDict


class OrderedDefaultDict(OrderedDict):
    def __missing__(self, key):
        self.__dict__[key] = list()
        return self.__dict__[key]

print(tabulate.tabulate_formats)

print(local['nvidia-smi']())

a = local['nvidia-smi']()

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

# infos = OrderedDict(sorted(infos.items()))
print(tabulate.tabulate(infos,  infos.keys(), tablefmt='fancy_grid'))
#print(tabulate.tabulate(infos,  infos.keys(), tablefmt='orgtbl'))
#print(tabulate.tabulate(infos,  infos.keys()))