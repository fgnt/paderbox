from pprint import pprint
from nt.utils.pynvml import *
import pwd
from contextlib import contextmanager

UID = 1
EUID = 2

brand_names = {
    NVML_BRAND_UNKNOWN: 'Unknown',
    NVML_BRAND_QUADRO: 'Quadro',
    NVML_BRAND_TESLA: 'Tesla',
    NVML_BRAND_NVS: 'NVS',
    NVML_BRAND_GRID: 'Grid',
    NVML_BRAND_GEFORCE: 'GeForce',
}


@contextmanager
def nvidia():
    try:
        yield nvmlInit()
    finally:
        nvmlShutdown()


def get_owner(pid):
    for ln in open('/proc/%d/status' % pid):
        if ln.startswith('Uid:'):
            uid = int(ln.split()[UID])
            return pwd.getpwuid(uid).pw_name


def get_info():
    with nvidia():
        info = dict(
            driver_version=nvmlSystemGetDriverVersion(),
            device_count=nvmlDeviceGetCount(),
        )
    return info


def get_gpu_list():
    info = get_info()
    with nvidia():
        gpu_list = []
        for index in range(info['device_count']):
            try:
                handle = nvmlDeviceGetHandleByIndex(index)
                pci_info = nvmlDeviceGetPciInfo(handle)
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                util_info = nvmlDeviceGetUtilizationRates(handle)
                processes = nvmlDeviceGetComputeRunningProcesses(handle)
                process_info = []
                for process in processes:
                    try:
                        name = nvmlSystemGetProcessName(process.pid)
                    except:
                        # Only report processes with names, the others are possibly
                        # zombie processes
                        continue
                    process_info.append(dict(
                        pid=process.pid,
                        name=name,
                        user=get_owner(process.pid),
                        memory=process.usedGpuMemory  # In bytes
                    ))
                gpu_list.append(dict(
                    bus_id=pci_info.busId,
                    name=nvmlDeviceGetName(handle),
                    product_brand=brand_names.get(nvmlDeviceGetBrand(handle)),
                    display_mode=bool(nvmlDeviceGetDisplayMode(handle)),
                    display_active=bool(nvmlDeviceGetDisplayActive(handle)),
                    persistence_mode=bool(nvmlDeviceGetPersistenceMode(handle)),
                    accounting_mode=bool(nvmlDeviceGetAccountingMode(handle)),
                    uuid=nvmlDeviceGetUUID(handle),
                    minor_number=nvmlDeviceGetMinorNumber(handle),
                    vbios_version=nvmlDeviceGetVbiosVersion(handle),
                    memory=dict(total=mem_info.total, used=mem_info.used),
                    utilization=dict(gpu=util_info.gpu, memory=util_info.memory),
                    temperature=nvmlDeviceGetTemperature(
                        handle, NVML_TEMPERATURE_GPU),
                    process_info=process_info
                ))
            except NVMLError as e:
                # ToDo: make a better print.
                print('ERROR in nt.utils.nvidia_helper.get_gpu_list for '
                      '{}: {}'.format(str(nvmlDeviceGetName(handle))[2:-1], e))
    return gpu_list


def print_processes():
    for gpu in get_gpu_list():
        print('GPU #{}'.format(gpu['minor_number']))
        pprint(gpu['process_info'])


if __name__ == "__main__":
    with nvidia():
        pprint(get_info())
        pprint(get_gpu_list())
