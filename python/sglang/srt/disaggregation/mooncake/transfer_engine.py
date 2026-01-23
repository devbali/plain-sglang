import json
import logging
import os
import socket
import time
from typing import List, Optional

from sglang.srt.environ import envs
from sglang.srt.disaggregation.common.utils import append_p2p_csv
from sglang.srt.utils import get_free_port, maybe_wrap_ipv6_address

logger = logging.getLogger(__name__)


def get_ib_devices_for_gpu(ib_device_str: Optional[str], gpu_id: int) -> Optional[str]:
    """
    Parse IB device string and get IB devices for a specific GPU ID.

    Supports all the following formats:
    1. Old format: "ib0, ib1, ib2"
    2. New format: {0: "ib0, ib1", 1: "ib2, ib3", 2: "ib4"}
    3. JSON file: path to a JSON file containing the mapping

    Args:
        ib_device_str: The original IB device string or path to JSON file
        gpu_id: The GPU ID to get devices for

    Returns:
        IB devices string for the GPU, or None if not available
    """
    if ib_device_str is None or not ib_device_str.strip():
        return None

    ib_device_str = ib_device_str.strip()

    # Check if it's a JSON file first and load its content
    is_json_file = ib_device_str.endswith(".json")
    if is_json_file:
        try:
            if os.path.isfile(ib_device_str):
                with open(ib_device_str, "r") as f:
                    ib_device_str = f.read()
            else:
                # File doesn't exist, treat as old format
                raise RuntimeError(f"File {ib_device_str} does not exist.")
        except (IOError, OSError) as e:
            # File reading failed, raise exception
            raise RuntimeError(f"Failed to read JSON file {ib_device_str}: {e}") from e

    # Check if it's JSON format (new format)
    try:
        parsed_json = json.loads(ib_device_str)
        if isinstance(parsed_json, dict):
            # Validate format - keys should be integers (or string rep), values should be strings
            gpu_mapping = {}
            for gpu_key, ib_devices in parsed_json.items():
                if (
                    isinstance(gpu_key, str)
                    and gpu_key.isdigit()
                    and isinstance(ib_devices, str)
                ):
                    gpu_mapping[int(gpu_key)] = ib_devices.strip()
                elif isinstance(gpu_key, int) and isinstance(ib_devices, str):
                    gpu_mapping[gpu_key] = ib_devices.strip()
                else:
                    raise ValueError(
                        f"Invalid format: keys must be integers (or string representations of integers) and values must be strings"
                    )

            if not gpu_mapping:
                raise ValueError("No valid GPU mappings found in JSON")

            # Return devices for specific GPU
            if gpu_id in gpu_mapping:
                return gpu_mapping[gpu_id]
            else:
                raise ValueError(
                    f"No IB devices configured for GPU {gpu_id}. Available GPUs: {list(gpu_mapping.keys())}"
                )

    except json.JSONDecodeError:
        if is_json_file:
            # It was supposed to be a JSON file but failed to parse
            raise RuntimeError(
                f"Failed to parse JSON content from file {ib_device_str}"
            )
        # Not JSON format, treat as old format - return same devices for all GPUs
        return ib_device_str


class MooncakeTransferEngine:

    def __init__(self, hostname: str, gpu_id: int, ib_device: Optional[str] = None):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run SGLang with MooncakeTransferEngine."
            ) from e

        self.engine = TransferEngine()
        self.hostname = hostname
        self.gpu_id = gpu_id
        self.ib_device = get_ib_devices_for_gpu(ib_device, gpu_id)
        logger.info(
            "Mooncake interface selection: ib_device=%s (raw=%s)",
            self.ib_device,
            ib_device,
        )
        resolved_ip = None
        try:
            resolved_ip = socket.gethostbyname(hostname)
        except Exception:
            resolved_ip = None
        logger.info(
            "MooncakeTransferEngine init: hostname=%s resolved_ip=%s gpu_id=%s ib_device=%s",
            self.hostname,
            resolved_ip,
            self.gpu_id,
            self.ib_device,
        )
        logger.info(
            "Mooncake env: NCCL_SOCKET_IFNAME=%s NCCL_IB_HCA=%s NCCL_IB_GID_INDEX=%s",
            os.getenv("NCCL_SOCKET_IFNAME"),
            os.getenv("NCCL_IB_HCA"),
            os.getenv("NCCL_IB_GID_INDEX"),
        )
        try:
            topo_filtered = self.engine.get_local_topology(device_name=self.ib_device or "")
            logger.info("Mooncake local topology (filtered=%s): %s", self.ib_device, topo_filtered)
            topo_all = self.engine.get_local_topology(device_name="")
            hca_names = self._extract_hcas_from_topology(topo_all)
            protocol = os.getenv("MC_PROTOCOL", "tcp")
            logger.info(
                "Mooncake transport selection: protocol=%s hcas=%s",
                protocol,
                ",".join(hca_names) if hca_names else "none",
            )
        except Exception as e:
            logger.warning("Mooncake local topology dump failed: %s", e)

        self.initialize(
            hostname=self.hostname,
            device_name=self.ib_device,
        )
        logger.info(
            "MooncakeTransferEngine initialized: hostname=%s gpu_id=%s device_name=%s rpc_port=%s",
            self.hostname,
            self.gpu_id,
            self.ib_device,
            self.engine.get_rpc_port(),
        )
        self.session_id = (
            f"{maybe_wrap_ipv6_address(self.hostname)}:{self.engine.get_rpc_port()}"
        )
        logger.info("MooncakeTransferEngine session_id=%s", self.session_id)

    @staticmethod
    def _extract_hcas_from_topology(topo_json: str) -> List[str]:
        try:
            topo = json.loads(topo_json)
        except Exception:
            return []
        hcas = set()
        for entry in topo.values():
            if not isinstance(entry, list) or len(entry) != 2:
                continue
            for group in entry:
                if not isinstance(group, list):
                    continue
                for hca in group:
                    if isinstance(hca, str) and hca:
                        hcas.add(hca)
        return sorted(hcas)

    def register(self, ptr, length):
        try:
            ret_value = self.engine.register_memory(ptr, length)
        except Exception:
            # Mark register as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug("Mooncake memory registration %s failed.", ptr)

    def deregister(self, ptr):
        try:
            ret_value = self.engine.unregister_memory(ptr)
        except Exception:
            # Mark deregister as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug("Mooncake memory deregistration %s failed.", ptr)

    def batch_register(self, ptrs: List[int], lengths: List[int]) -> int:
        """Batch register multiple memory regions."""
        try:
            ret_value = self.engine.batch_register_memory(ptrs, lengths)
        except Exception:
            # Mark batch register as failed
            ret_value = -1
            if not hasattr(self.engine, "batch_register_memory"):
                raise RuntimeError(
                    "Mooncake's batch register requires a newer version of mooncake-transfer-engine. "
                    "Please upgrade Mooncake."
                )

        if ret_value != 0:
            logger.debug("Mooncake batch memory registration failed.")
        return ret_value

    def batch_deregister(self, ptrs: List[int]) -> int:
        """Batch deregister multiple memory regions."""
        try:
            ret_value = self.engine.batch_unregister_memory(ptrs)
        except Exception:
            # Mark batch deregister as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug("Mooncake batch memory deregistration failed.")
        return ret_value

    def initialize(
        self,
        hostname: str,
        device_name: Optional[str],
    ) -> None:
        """Initialize the mooncake instance."""
        logger.info(
            "MooncakeTransferEngine initialize: hostname=%s device_name=%s",
            hostname,
            device_name,
        )
        if envs.ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE.get():
            npu_phy_id = envs.ASCEND_NPU_PHY_ID.get()
            if npu_phy_id == -1:
                hostname += f":{get_free_port()}:npu_{self.gpu_id}"
            else:
                hostname += f":{get_free_port()}:npu_{npu_phy_id}"
            ret_value = self.engine.initialize(
                hostname,
                "P2PHANDSHAKE",
                "ascend",
                device_name if device_name is not None else "",
            )
        else:
            ret_value = self.engine.initialize(
                hostname,
                "P2PHANDSHAKE",
                "rdma",
                device_name if device_name is not None else "",
            )
        if ret_value != 0:
            logger.error("Mooncake Transfer Engine initialization failed.")
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        """Synchronously transfer data to the specified address."""
        logger.info(
            "Mooncake transfer_sync: session_id=%s buffer=0x%x peer=0x%x len=%s",
            session_id,
            buffer,
            peer_buffer_address,
            length,
        )
        start = time.perf_counter()
        try:
            # the first time: based on session_id (which contains remote_ip) to construct a queue pair, and cache the queue pair
            # later: based on the cached queue pair to send data
            ret = self.engine.transfer_sync_write(
                session_id, buffer, peer_buffer_address, length
            )
        except Exception:
            # Mark transfer request as failed
            ret = -1

        if ret < 0:
            # Do not raise an exception here, since some transfer requests fail should be accepted and the execution thread should not be stopped.
            logger.debug(
                "Failed to transfer data from %s to %s - %s.",
                buffer,
                session_id,
                peer_buffer_address,
            )

        append_p2p_csv("p2p_data.csv", length, time.perf_counter() - start)
        return ret

    def batch_transfer_sync(
        self,
        session_id: str,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
    ) -> int:
        """Synchronously transfer data to the specified addresses in batches."""
        logger.info(
            "Mooncake batch_transfer_sync: session_id=%s count buffers=%s, lengths of each buffer=%s",
            session_id,
            len(buffers),
            str(lengths)
        )
        start = time.perf_counter()
        try:
            ret = self.engine.batch_transfer_sync_write(
                session_id, buffers, peer_buffer_addresses, lengths
            )
        except Exception:
            ret = -1
            # Inform user to upgrade mooncake-transfer-engine >= 0.3.4.post2
            if not hasattr(self.engine, "batch_transfer_sync_write"):
                raise RuntimeError(
                    "Mooncake's batch transfer requires mooncake-transfer-engine >= 0.3.4.post2. "
                    "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'"
                )

        if ret < 0:
            logger.debug(
                "Failed to batch transfer data. Buffers: %s, Session: %s, Peer addresses: %s",
                buffers,
                session_id,
                peer_buffer_addresses,
            )
        append_p2p_csv("p2p_data.csv", sum(lengths), time.perf_counter() - start)
        return ret

    def get_session_id(self):
        return self.session_id
