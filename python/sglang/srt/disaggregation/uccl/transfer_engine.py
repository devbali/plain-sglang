import logging
import os
import socket
import struct
import threading
from typing import List, Optional, Tuple

from sglang.srt.utils import maybe_wrap_ipv6_address

logger = logging.getLogger(__name__)


class UcclTransferEngine:
    _UCCL_FIFO_ADDR_OFFSET = 0
    _UCCL_FIFO_SIZE_OFFSET = 8

    def __init__(
        self,
        hostname: str,
        gpu_id: int,
        num_cpus: Optional[int] = None,
    ):
        if "UCCL_RCMODE" not in os.environ:
            os.environ["UCCL_RCMODE"] = "1"
            logger.info("UCCL env: defaulting UCCL_RCMODE=1 for RC mode")
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError("PyTorch is required to use UCCL.") from e

        try:
            from uccl import p2p
        except ImportError as e:
            raise ImportError(
                "Please install UCCL by following the instructions at "
                "https://github.com/uccl-project/uccl/tree/main/p2p "
                "to run SGLang with UcclTransferEngine."
            ) from e

        self._p2p = p2p
        if num_cpus is None:
            num_cpus = 1 # Default to 1 CPU if not specified

        logger.info(
            "UCCL env: UCCL_SOCKET_IFNAME=%s NCCL_SOCKET_IFNAME=%s UCCL_IB_HCA=%s UCCL_IB_GID_INDEX=%s",
            os.getenv("UCCL_SOCKET_IFNAME"),
            os.getenv("NCCL_SOCKET_IFNAME"),
            os.getenv("UCCL_IB_HCA"),
            os.getenv("UCCL_IB_GID_INDEX"),
        )
        ifname = os.getenv("UCCL_SOCKET_IFNAME")
        ifname_ip = None
        if ifname:
            ifname_ip = self._get_ipv4_for_ifname(ifname)
            logger.info("UCCL socket ifname %s resolves to ip=%s", ifname, ifname_ip)
        else:
            auto_ifname = self._select_ifname_from_hca()
            if auto_ifname:
                ifname = auto_ifname
                ifname_ip = self._get_ipv4_for_ifname(ifname)
                logger.info(
                    "UCCL auto-selected ifname %s from HCA=%s GID_INDEX=%s ip=%s",
                    ifname,
                    os.getenv("UCCL_IB_HCA"),
                    os.getenv("UCCL_IB_GID_INDEX"),
                    ifname_ip,
                )
        self._ifname_ip = ifname_ip
        logger.info("num cpus for UCCL: %s", num_cpus   )
        self.endpoint = p2p.Endpoint(gpu_id, num_cpus)
        self.hostname = hostname
        self.gpu_id = gpu_id

        self._metadata = bytes(self.endpoint.get_metadata())
        meta_ip, meta_port, meta_gpu = self._p2p.Endpoint.parse_metadata(self._metadata)
        if self._ifname_ip and len(self._metadata) == 10:
            self._session_id = f"{maybe_wrap_ipv6_address(ifname_ip)}:{meta_port}"
            logger.info(
                "UCCL session_id override: metadata_ip=%s -> ifname_ip=%s",
                meta_ip,
                ifname_ip,
            )
        else:
            self._session_id = f"{maybe_wrap_ipv6_address(meta_ip)}:{meta_port}"
        self._conn_by_peer = {}
        self._conn_by_ip = {}
        self._conn_lock = threading.Lock()
        logger.info(
            "UCCL local metadata parse: ip=%s port=%s gpu=%s len=%s",
            meta_ip,
            meta_port,
            meta_gpu,
            len(self._metadata),
        )

    @staticmethod
    def _get_ipv4_for_ifname(ifname: str) -> Optional[str]:
        try:
            import fcntl
        except Exception:
            return None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            ifreq = struct.pack("256s", ifname[:15].encode("ascii"))
            res = fcntl.ioctl(sock.fileno(), 0x8915, ifreq)  # SIOCGIFADDR
            return socket.inet_ntoa(res[20:24])
        except Exception:
            return None

    @staticmethod
    def _read_gid_ndev(hca: str, port: int, gid_index: int) -> Optional[str]:
        path = f"/sys/class/infiniband/{hca}/ports/{port}/gid_attrs/ndevs/{gid_index}"
        try:
            with open(path, "r") as f:
                ndev = f.read().strip()
                return ndev or None
        except Exception:
            return None

    def _select_ifname_from_hca(self) -> Optional[str]:
        hca = os.getenv("UCCL_IB_HCA")
        if not hca:
            return None
        gid_env = os.getenv("UCCL_IB_GID_INDEX")
        gid_index = int(gid_env) if gid_env is not None else None
        ports = [1]
        try:
            ports_dir = f"/sys/class/infiniband/{hca}/ports"
            ports = [
                int(p)
                for p in os.listdir(ports_dir)
                if p.isdigit()
            ] or ports
        except Exception:
            pass
        if gid_index is not None:
            for port in ports:
                ndev = self._read_gid_ndev(hca, port, gid_index)
                if ndev:
                    return ndev
            return None
        for port in ports:
            gid_attrs = f"/sys/class/infiniband/{hca}/ports/{port}/gid_attrs/ndevs"
            try:
                for entry in sorted(os.listdir(gid_attrs)):
                    if not entry.isdigit():
                        continue
                    ndev = self._read_gid_ndev(hca, port, int(entry))
                    if ndev:
                        return ndev
            except Exception:
                continue
        return None

    def _override_metadata_ip(self, metadata: bytes, ip: Optional[str]) -> bytes:
        if not ip:
            return metadata
        if len(metadata) != 10:
            logger.info("UCCL metadata override skipped (non-IPv4 len=%s)", len(metadata))
            return metadata
        try:
            ip_bytes = socket.inet_pton(socket.AF_INET, ip)
        except OSError:
            logger.info("UCCL metadata override skipped (invalid ip=%s)", ip)
            return metadata
        buf = bytearray(metadata)
        buf[0:4] = ip_bytes
        return bytes(buf)

    def get_metadata_for_send(self) -> bytes:
        if_ip = self._ifname_ip
        if not if_ip:
            return self._metadata
        overridden = self._override_metadata_ip(self._metadata, if_ip)
        if overridden != self._metadata:
            logger.info(
                "UCCL metadata override for send: ip=%s",
                if_ip,
            )
        return overridden

    def start_accept_thread(self):
        def accept_loop():
            while True:
                try:
                    success, remote_ip, remote_gpu_idx, conn_id = self.endpoint.accept()
                    if not success:
                        continue
                    logger.info(
                        "UCCL accepted connection from %s gpu=%s conn_id=%s",
                        remote_ip,
                        remote_gpu_idx,
                        conn_id,
                    )
                    with self._conn_lock:
                        self._conn_by_peer[(remote_ip, remote_gpu_idx)] = conn_id
                        self._conn_by_ip[remote_ip] = conn_id
                except Exception as e:
                    logger.error("UCCL accept thread failed: %s", e)

        threading.Thread(target=accept_loop, daemon=True).start()

    def _parse_peer(self, session_id: str) -> Tuple[str, int, Optional[int]]:
        if session_id.startswith("["):
            addr, port_str = session_id.split("]:", 1)
            return addr[1:], int(port_str), None
        if "." in session_id:
            ip, port_str = session_id.rsplit(":", 1)
            return ip, int(port_str), None
        metadata = session_id.encode("ascii")
        try:
            ip, port, gpu_idx = self._p2p.Endpoint.parse_metadata(metadata)
            return ip, port, gpu_idx
        except Exception:
            return session_id, 0, None

    def connect_session(self, session_id: str) -> Optional[int]:
        ip, port, gpu_idx = self._parse_peer(session_id)
        if gpu_idx is None:
            logger.error("UCCL connect_session missing gpu_idx for %s", session_id)
            return None
        try:
            success, conn_id = self.endpoint.connect(ip, gpu_idx, remote_port=port)
        except Exception:
            success, conn_id = False, None
        if not success:
            logger.error("UCCL connect failed for %s", session_id)
            return None
        with self._conn_lock:
            self._conn_by_peer[(ip, gpu_idx)] = conn_id
            self._conn_by_ip[ip] = conn_id
        return conn_id

    def connect_metadata(self, metadata: bytes) -> Optional[int]:
        logger.info("UCCL connect_metadata enter (metadata_len=%s)", len(metadata))
        ip, port, gpu_idx = self._p2p.Endpoint.parse_metadata(metadata)
        logger.info("UCCL connect to %s gpu=%s port=%s", ip, gpu_idx, port)
        conn_id = self._connect(ip, gpu_idx, port)
        logger.info("UCCL connect_metadata result: conn_id=%s", conn_id)
        return conn_id

    def connect_endpoint(self, ip: str, metadata: bytes) -> Optional[int]:
        logger.info(
            "UCCL connect_endpoint enter: prefill_ip=%s metadata_len=%s",
            ip,
            len(metadata),
        )
        metadata_ip, port, gpu_idx = self._p2p.Endpoint.parse_metadata(metadata)
        logger.info(
            "UCCL metadata parse: ip=%s port=%s gpu=%s metadata_len=%s",
            metadata_ip,
            port,
            gpu_idx,
            len(metadata),
        )
        logger.info(
            "UCCL connect to %s (prefill_ip=%s) gpu=%s port=%s",
            metadata_ip,
            ip,
            gpu_idx,
            port,
        )
        conn_id = self._connect(metadata_ip, gpu_idx, port)
        logger.info("UCCL connect_endpoint result: conn_id=%s", conn_id)
        return conn_id

    def parse_metadata(self, metadata: bytes) -> Tuple[str, int, int]:
        return self._p2p.Endpoint.parse_metadata(metadata)

    def _connect(self, ip: str, gpu_idx: int, port: int) -> Optional[int]:
        logger.info(
            "UCCL _connect enter: ip=%s gpu=%s port=%s", ip, gpu_idx, port
        )
        try:
            success, conn_id = self.endpoint.connect(ip, gpu_idx, remote_port=port)
        except Exception:
            success, conn_id = False, None
        if not success:
            logger.error("UCCL connect failed for %s", ip)
            return None
        logger.info("UCCL _connect ok: ip=%s gpu=%s conn_id=%s", ip, gpu_idx, conn_id)
        with self._conn_lock:
            self._conn_by_peer[(ip, gpu_idx)] = conn_id
            self._conn_by_ip[ip] = conn_id
        return conn_id

    def get_conn_id(self, session_id: str) -> Optional[int]:
        ip, _port, gpu_idx = self._parse_peer(session_id)
        with self._conn_lock:
            if gpu_idx is not None:
                conn_id = self._conn_by_peer.get((ip, gpu_idx))
                if conn_id is not None:
                    return conn_id
            return self._conn_by_ip.get(ip)

    def _update_fifo_item(self, base_meta: bytes, addr: int, size: int) -> bytes:
        buf = bytearray(base_meta)
        struct.pack_into("<Q", buf, self._UCCL_FIFO_ADDR_OFFSET, addr)
        struct.pack_into("<I", buf, self._UCCL_FIFO_SIZE_OFFSET, size)
        return bytes(buf)

    def batch_register(self, ptrs: List[int], lengths: List[int]) -> List[int]:
        if not ptrs:
            return []
        try:
            ok, mr_ids = self.endpoint.regv(ptrs, lengths)
        except Exception:
            ok, mr_ids = False, []
        if not ok:
            logger.error("UCCL batch register failed.")
            return []
        return list(mr_ids)

    def batch_deregister(self, mr_ids: List[int]) -> int:
        if not mr_ids:
            return 0
        failure = 0
        for mr_id in mr_ids:
            try:
                ok = self.endpoint.dereg(mr_id)
            except Exception:
                ok = False
            if not ok:
                failure += 1
        return 0 if failure == 0 else -1

    def batch_advertise(
        self,
        conn_id: int,
        ptrs: List[int],
        lengths: List[int],
        mr_ids: List[int],
    ) -> List[bytes]:
        if not ptrs:
            return []
        logger.info(
            "UCCL batch_advertise enter: conn_id=%s count=%s",
            conn_id,
            len(ptrs),
        )
        try:
            ok, meta_list = self.endpoint.advertisev(
                conn_id, mr_ids, ptrs, lengths, len(ptrs)
            )
        except Exception:
            ok, meta_list = False, []
        if not ok:
            logger.error("UCCL advertise failed.")
            return []
        logger.info("UCCL batch_advertise ok: conn_id=%s count=%s", conn_id, len(meta_list))
        return [bytes(meta) for meta in meta_list]

    def batch_transfer_sync(
        self,
        session_id: str,
        src_addrs: List[int],
        dst_addrs: List[int],
        lengths: List[int],
        src_mr_ids: List[int],
        dst_base_ptrs: List[int],
        dst_metas: List[bytes],
    ) -> int:
        conn_id = self.get_conn_id(session_id)
        if conn_id is None:
            logger.error("UCCL session %s is not connected.", session_id)
            return -1

        logger.info(
            "UCCL batch_transfer_sync: session=%s conn_id=%s blocks=%d block lengths=%s",
            session_id,
            conn_id,
            len(src_addrs),
            str(lengths),
        )
        if not (
            len(src_addrs)
            == len(dst_addrs)
            == len(lengths)
            == len(src_mr_ids)
            == len(dst_base_ptrs)
            == len(dst_metas)
        ):
            logger.error(
                "UCCL batch_transfer_sync: mismatched list sizes: "
                "src_addrs=%d dst_addrs=%d lengths=%d src_mr_ids=%d dst_base_ptrs=%d dst_metas=%d",
                len(src_addrs),
                len(dst_addrs),
                len(lengths),
                len(src_mr_ids),
                len(dst_base_ptrs),
                len(dst_metas),
            )
            return -1

        meta_list = []
        for dst_addr, length, dst_base_ptr, dst_meta in zip(
            dst_addrs, lengths, dst_base_ptrs, dst_metas
        ):
            if length > 0xFFFFFFFF:
                logger.error("UCCL transfer size exceeds 4GB: %s", length)
                return -1
            offset = dst_addr - dst_base_ptr
            meta_list.append(
                self._update_fifo_item(dst_meta, dst_base_ptr + offset, length)
            )

        try:
            logger.info("UCCL batch_transfer_sync: starting writev")
            ok = self.endpoint.writev(
                conn_id, src_mr_ids, src_addrs, lengths, meta_list, len(src_addrs)
            )
            logger.info("UCCL batch_transfer_sync: finished writev")

        except Exception:
            ok = False
        return 0 if ok else -1

    def get_session_id(self) -> str:
        return self._session_id

    def get_metadata(self) -> bytes:
        return self._metadata
