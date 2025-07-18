import logging

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh


def squeeze_to_rank(x: jax.Array, rank: int):
    tmp = x
    while tmp.ndim > rank:
        tmp = tmp.squeeze(0)

    return tmp


def expand_to_rank(x: jax.Array, rank: int):
    tmp = x
    while tmp.ndim != rank:
        tmp = jnp.expand_dims(tmp, axis=0)

    return tmp


def create_mesh(mesh_shape: tuple[int, ...], axis_names: tuple[str, ...]):
    devices = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(devices, axis_names=axis_names)
    return mesh


def log_node_devices_stats(logger: logging.Logger):
    devices = jax.devices()
    logger.info(f"Found {len(devices)} devices.")

    for device in devices:
        try:
            stats = device.memory_stats()
            # Collect memory stats in a compact format
            stat_items = []
            for key, value in stats.items():
                if "bytes" in key:
                    value_gb = value / (1024**3)
                    stat_items.append(f"{key}: {value_gb:.2f} GB")
                else:
                    stat_items.append(f"{key}: {value}")
            logger.info(f"{device}: " + "; ".join(stat_items))
        except Exception as e:
            logger.warning(f"Could not get memory stats for {device}: {e}")
