import json
import os
import socket
from pathlib import Path

import ray
from ray.util.placement_group import placement_group, PlacementGroupSchedulingStrategy


def snapshot(label: str):
    import os
    import subprocess
    import ray
    import torch
    import verl.utils.device as device_utils

    try:
        nvidia_smi = subprocess.run(
            ['bash', '-lc', 'nvidia-smi -L 2>/dev/null | sed -n "1,4p"'],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except Exception as exc:
        nvidia_smi = f'ERR:{exc!r}'

    return {
        'label': label,
        'host': socket.gethostname(),
        'pid': os.getpid(),
        'cwd': os.getcwd(),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'NVIDIA_VISIBLE_DEVICES': os.environ.get('NVIDIA_VISIBLE_DEVICES'),
        'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': os.environ.get('RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES'),
        'RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO': os.environ.get('RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO'),
        'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH'),
        'has_dev_nvidia0': Path('/dev/nvidia0').exists(),
        'has_dev_nvidiactl': Path('/dev/nvidiactl').exists(),
        'nvidia_smi_L': nvidia_smi,
        'ray_gpu_ids': ray.get_gpu_ids() if ray.is_initialized() else [],
        'assigned_resources': (
            dict(ray.get_runtime_context().get_assigned_resources())
            if ray.get_runtime_context().worker.mode == ray._private.worker.WORKER_MODE
            else {}
        ),
        'torch_version_cuda': torch.version.cuda,
        'torch_cuda_available': torch.cuda.is_available(),
        'torch_cuda_device_count': torch.cuda.device_count(),
        'device_module_is_cuda_available': device_utils.is_cuda_available,
        'device_module_get_device_name': device_utils.get_device_name(),
        'device_module_visible_keyword': device_utils.get_visible_devices_keyword(),
    }


@ray.remote(num_gpus=1 / 3)
class FractionalProbe:
    def info(self, label: str):
        return snapshot(label)


@ray.remote(num_gpus=1)
class FullProbe:
    def info(self, label: str):
        return snapshot(label)


def main():
    common_env = {
        'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': '1',
        'RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO': '0',
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'NVIDIA_VISIBLE_DEVICES': os.environ.get('NVIDIA_VISIBLE_DEVICES', 'all'),
    }
    ray.init(address=os.environ['RAY_ADDRESS'], runtime_env={'env_vars': common_env})

    results = [snapshot('driver')]
    results.append(ray.get(FullProbe.remote().info.remote('plain_full')))
    results.append(ray.get(FractionalProbe.remote().info.remote('plain_fractional')))

    pg = placement_group([{'CPU': 3, 'GPU': 1}] * 4, strategy='STRICT_PACK')
    ray.get(pg.ready())

    probe_runtime_env = {
        'env_vars': {
            'WORLD_SIZE': '4',
            'RANK': '0',
            'WG_PREFIX': 'probe',
            'WG_BACKEND': 'ray',
            'RAY_LOCAL_WORLD_SIZE': '4',
            'MASTER_ADDR': '127.0.0.1',
            'MASTER_PORT': '29500',
            **common_env,
        }
    }

    fractional_pg_actor = FractionalProbe.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=0),
        runtime_env=probe_runtime_env,
        name='probe_fractional_pg',
    ).remote()
    results.append(ray.get(fractional_pg_actor.info.remote('pg_fractional')))

    full_pg_actor = FullProbe.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=1),
        runtime_env=probe_runtime_env,
        name='probe_full_pg',
    ).remote()
    results.append(ray.get(full_pg_actor.info.remote('pg_full')))

    print(json.dumps(results, indent=2, sort_keys=True))
    ray.shutdown()


if __name__ == '__main__':
    main()
