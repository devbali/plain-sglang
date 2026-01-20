from __future__ import annotations

import concurrent.futures
import logging
import os
import struct
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt
import requests

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    group_concurrent_contiguous,
)
from sglang.srt.disaggregation.mooncake.conn import (
    AuxDataCodec,
    KVArgsRegisterInfo,
    KVTransferError,
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
    TransferInfo,
    TransferKVChunk,
)
from sglang.srt.disaggregation.uccl.transfer_engine import UcclTransferEngine
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import format_tcp_address, get_int_env_var, is_valid_ipv6_address

logger = logging.getLogger(__name__)


class UcclKVArgsRegisterInfo(KVArgsRegisterInfo):
    def __init__(
        self,
        room: str,
        endpoint: str,
        dst_port: int,
        mooncake_session_id: str,
        dst_kv_ptrs: list[int],
        dst_aux_ptrs: list[int],
        dst_state_data_ptrs: list[int],
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        dst_kv_metas: Optional[List[bytes]] = None,
        dst_aux_metas: Optional[List[bytes]] = None,
        dst_state_metas: Optional[List[bytes]] = None,
        meta_ready: Optional[threading.Event] = None,
        connection_ready: Optional[threading.Event] = None,
    ):
        super().__init__(
            room=room,
            endpoint=endpoint,
            dst_port=dst_port,
            mooncake_session_id=mooncake_session_id,
            dst_kv_ptrs=dst_kv_ptrs,
            dst_aux_ptrs=dst_aux_ptrs,
            dst_state_data_ptrs=dst_state_data_ptrs,
            dst_tp_rank=dst_tp_rank,
            dst_attn_tp_size=dst_attn_tp_size,
            dst_kv_item_len=dst_kv_item_len,
        )
        self.dst_kv_metas = dst_kv_metas
        self.dst_aux_metas = dst_aux_metas
        self.dst_state_metas = dst_state_metas
        self.meta_ready = meta_ready or threading.Event()
        self.connection_ready = connection_ready or threading.Event()

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[5])//8}Q", msg[5])),
            dst_state_data_ptrs=list(struct.unpack(f"{len(msg[6])//8}Q", msg[6])),
            dst_tp_rank=int(msg[7].decode("ascii")),
            dst_attn_tp_size=int(msg[8].decode("ascii")),
            dst_kv_item_len=int(msg[9].decode("ascii")),
        )


class UcclKVManager(MooncakeKVManager):
    AUX_DATA_HEADER = b"AUX_DATA"
    UCCL_ENDPOINT_HEADER = b"UCCL_ENDPOINT"
    UCCL_META_HEADER = b"UCCL_META"

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        self.ptr_to_mr_id: Dict[int, int] = {}
        self.kv_mr_ids: List[int] = []
        self.aux_mr_ids: List[int] = []
        self.state_mr_ids: List[int] = []
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.enable_custom_mem_pool = False
        self.custom_mem_pool_type = None

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.engine.start_accept_thread()

    def init_engine(self):
        self.engine = UcclTransferEngine(
            hostname=self.local_ip,
            gpu_id=self.kv_args.gpu_id,
        )

    def register_buffer_to_engine(self):
        if self.kv_args.kv_data_ptrs and self.kv_args.kv_data_lens:
            self.kv_mr_ids = self.engine.batch_register(
                self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
            )
            for ptr, mr_id in zip(self.kv_args.kv_data_ptrs, self.kv_mr_ids):
                self.ptr_to_mr_id[ptr] = mr_id

        if self.kv_args.aux_data_ptrs and self.kv_args.aux_data_lens:
            self.aux_mr_ids = self.engine.batch_register(
                self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
            )
            for ptr, mr_id in zip(self.kv_args.aux_data_ptrs, self.aux_mr_ids):
                self.ptr_to_mr_id[ptr] = mr_id

        if self.kv_args.state_data_ptrs and self.kv_args.state_data_lens:
            self.state_mr_ids = self.engine.batch_register(
                self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
            )
            for ptr, mr_id in zip(self.kv_args.state_data_ptrs, self.state_mr_ids):
                self.ptr_to_mr_id[ptr] = mr_id

    def _transfer_data(
        self,
        mooncake_session_id: str,
        transfer_blocks: List[Tuple[int, int, int, int, int, bytes]],
    ):
        if not transfer_blocks:
            return 0

        (
            src_addrs,
            dst_addrs,
            lengths,
            src_mr_ids,
            dst_base_ptrs,
            dst_metas,
        ) = zip(*transfer_blocks)
        logger.info(
            "UCCL transfer: session=%s blocks=%d unique_src_mr_ids=%d",
            mooncake_session_id,
            len(transfer_blocks),
            len(set(src_mr_ids)),
        )
        return self.engine.batch_transfer_sync(
            mooncake_session_id,
            list(src_addrs),
            list(dst_addrs),
            list(lengths),
            list(src_mr_ids),
            list(dst_base_ptrs),
            list(dst_metas),
        )

    def _send_kvcache_generic(
        self,
        mooncake_session_id: str,
        src_data_ptrs: list[int],
        dst_data_ptrs: list[int],
        item_lens: list[int],
        prefill_data_indices: npt.NDArray[np.int32],
        dst_data_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ) -> int:
        dst_data_metas = self._get_uccl_metas(mooncake_session_id, dst_data_ptrs)
        if not dst_data_metas or len(dst_data_metas) != len(dst_data_ptrs):
            logger.error(
                "UCCL metadata is missing or mismatched for session %s",
                mooncake_session_id,
            )
            return -1

        logger.info(
            "UCCL send_kvcache_generic: session=%s layers=%d indices=%d",
            mooncake_session_id,
            len(item_lens),
            len(prefill_data_indices),
        )
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_data_indices, dst_data_indices
        )

        layers_params = None
        dst_meta_map = {
            ptr: meta for ptr, meta in zip(dst_data_ptrs, dst_data_metas or [])
        }

        if self.is_mla_backend:
            src_kv_ptrs, dst_kv_ptrs, layers_current_pp_stage = (
                self.get_mla_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs)
            )
            layers_params = [
                (
                    src_kv_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    item_lens[layer_id],
                    self.ptr_to_mr_id[src_kv_ptrs[layer_id]],
                    dst_meta_map[dst_kv_ptrs[layer_id]],
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        else:
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
                self.get_mha_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs)
            )
            layers_params = [
                (
                    src_k_ptrs[layer_id],
                    dst_k_ptrs[layer_id],
                    item_lens[layer_id],
                    self.ptr_to_mr_id[src_k_ptrs[layer_id]],
                    dst_meta_map[dst_k_ptrs[layer_id]],
                )
                for layer_id in range(layers_current_pp_stage)
            ] + [
                (
                    src_v_ptrs[layer_id],
                    dst_v_ptrs[layer_id],
                    item_lens[layers_current_pp_stage + layer_id],
                    self.ptr_to_mr_id[src_v_ptrs[layer_id]],
                    dst_meta_map[dst_v_ptrs[layer_id]],
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        assert layers_params is not None

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, item_len: int, src_mr_id: int, dst_meta: bytes
        ) -> List[Tuple[int, int, int, int, int, bytes]]:
            transfer_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append(
                    (src_addr, dst_addr, length, src_mr_id, dst_ptr, dst_meta)
                )
            return transfer_blocks

        def process_layer(
            src_ptr: int,
            dst_ptr: int,
            item_len: int,
            src_mr_id: int,
            dst_meta: bytes,
        ) -> int:
            transfer_blocks = set_transfer_blocks(
                src_ptr, dst_ptr, item_len, src_mr_id, dst_meta
            )
            return self._transfer_data(
                mooncake_session_id,
                transfer_blocks,
            )

        def process_layers(layers_params: List[Tuple[int, int, int, int, bytes]]) -> int:
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len, src_mr_id, dst_meta in layers_params:
                transfer_blocks.extend(
                    set_transfer_blocks(src_ptr, dst_ptr, item_len, src_mr_id, dst_meta)
                )
            if not transfer_blocks:
                return 0
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        if self.enable_custom_mem_pool:
            futures = [
                executor.submit(
                    process_layer,
                    src_ptr,
                    dst_ptr,
                    item_len,
                    src_mr_id,
                    dst_meta,
                )
                for (src_ptr, dst_ptr, item_len, src_mr_id, dst_meta) in layers_params
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for f in futures:
                        f.cancel()
                    return status
            return 0

        return process_layers(layers_params)

        return 0

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        logger.info(
            "UCCL send_kvcache: session=%s prefill_indices=%d dst_ptrs=%d",
            mooncake_session_id,
            len(prefill_kv_indices),
            len(dst_kv_ptrs),
        )
        return super().send_kvcache(
            mooncake_session_id,
            prefill_kv_indices,
            dst_kv_ptrs,
            dst_kv_indices,
            executor,
        )

    def send_kvcache_slice(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int64],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int64],
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        dst_kv_metas = self._get_uccl_metas(mooncake_session_id, dst_kv_ptrs)
        if not dst_kv_metas or len(dst_kv_metas) != len(dst_kv_ptrs):
            logger.error("UCCL metadata is missing or mismatched for session %s", mooncake_session_id)
            return -1

        logger.info(
            "UCCL send_kvcache_slice: session=%s prefill_indices=%d dst_ptrs=%d",
            mooncake_session_id,
            len(prefill_kv_indices),
            len(dst_kv_ptrs),
        )
        local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
        src_kv_item_len = self.kv_args.kv_item_lens[0]
        dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size
        num_kv_heads = self.kv_args.kv_head_num
        page_size = self.kv_args.page_size

        src_heads_per_rank = num_kv_heads
        dst_heads_per_rank = num_kv_heads * self.attn_tp_size // dst_attn_tp_size
        bytes_per_head_slice_to_send = (
            dst_kv_item_len // page_size // dst_heads_per_rank
        )

        if self.attn_tp_size > dst_attn_tp_size:
            src_head_start_offset = 0
            num_heads_to_send = src_heads_per_rank
            dst_head_start_offset = local_tp_rank_in_group * src_heads_per_rank
        else:
            src_head_start_offset = (
                dst_tp_rank_in_group * dst_heads_per_rank
            ) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start_offset = 0

        src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
            self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
        )
        dst_meta_map = {ptr: meta for ptr, meta in zip(dst_kv_ptrs, dst_kv_metas)}

        src_head_slice_offset = src_head_start_offset * bytes_per_head_slice_to_send
        dst_head_slice_offset = dst_head_start_offset * bytes_per_head_slice_to_send
        heads_bytes_per_token_to_send = num_heads_to_send * bytes_per_head_slice_to_send

        if heads_bytes_per_token_to_send > (dst_kv_item_len // page_size):
            logger.error(
                f"[{mooncake_session_id}] slice size ({heads_bytes_per_token_to_send}) exceeds "
                f"target token slot size ({dst_kv_item_len // page_size})"
            )
            return -1

        layers_params = [
            (
                src_k_ptrs[layer_id],
                dst_k_ptrs[layer_id],
                src_kv_item_len,
                dst_kv_item_len,
                src_head_slice_offset,
                dst_head_slice_offset,
                heads_bytes_per_token_to_send,
                self.ptr_to_mr_id[src_k_ptrs[layer_id]],
                dst_meta_map[dst_k_ptrs[layer_id]],
            )
            for layer_id in range(layers_current_pp_stage)
        ] + [
            (
                src_v_ptrs[layer_id],
                dst_v_ptrs[layer_id],
                src_kv_item_len,
                dst_kv_item_len,
                src_head_slice_offset,
                dst_head_slice_offset,
                heads_bytes_per_token_to_send,
                self.ptr_to_mr_id[src_v_ptrs[layer_id]],
                dst_meta_map[dst_v_ptrs[layer_id]],
            )
            for layer_id in range(layers_current_pp_stage)
        ]

        def process_layer_tp_aware(layer_params):
            (
                src_ptr,
                dst_ptr,
                src_item_len,
                dst_item_len,
                src_head_slice_offset,
                dst_head_slice_offset,
                heads_bytes_per_token_to_send,
                src_mr_id,
                dst_meta,
            ) = layer_params
            transfer_blocks = []

            bytes_per_token_on_prefill = src_item_len // page_size
            bytes_per_token_on_decode = dst_item_len // page_size

            for i in range(len(prefill_kv_indices)):
                prefill_page_idx = int(prefill_kv_indices[i])
                decode_page_idx = int(dst_kv_indices[i])

                src_page_start_addr = src_ptr + prefill_page_idx * src_item_len
                dst_page_start_addr = dst_ptr + decode_page_idx * dst_item_len

                for token_slot_in_page in range(page_size):
                    src_token_slot_start_addr = (
                        src_page_start_addr
                        + token_slot_in_page * bytes_per_token_on_prefill
                    )
                    dst_token_slot_start_addr = (
                        dst_page_start_addr
                        + token_slot_in_page * bytes_per_token_on_decode
                    )

                    src_slice_addr = src_token_slot_start_addr + src_head_slice_offset
                    dst_slice_addr = dst_token_slot_start_addr + dst_head_slice_offset

                    transfer_blocks.append(
                        (
                            src_slice_addr,
                            dst_slice_addr,
                            heads_bytes_per_token_to_send,
                            src_mr_id,
                            dst_ptr,
                            dst_meta,
                        )
                    )

            return self._transfer_data(
                mooncake_session_id,
                transfer_blocks,
            )

        futures = [
            executor.submit(
                process_layer_tp_aware,
                layer_params,
            )
            for layer_params in layers_params
        ]

        for future in concurrent.futures.as_completed(futures):
            status = future.result()
            if status != 0:
                for f in futures:
                    f.cancel()
                return status

        return 0

    def send_aux(
        self,
        req: TransferInfo,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
    ):
        if not dst_aux_ptrs:
            return 0
        dst_aux_metas = self._get_uccl_metas(req.mooncake_session_id, dst_aux_ptrs)
        if len(dst_aux_metas) != len(dst_aux_ptrs):
            logger.error("UCCL metadata is missing or mismatched for aux transfer")
            return -1

        logger.info(
            "UCCL send_aux: session=%s dst_ptrs=%d",
            req.mooncake_session_id,
            len(dst_aux_ptrs),
        )
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens
        transfer_blocks = []

        for i, dst_aux_ptr in enumerate(dst_aux_ptrs):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            dst_addr = dst_aux_ptrs[i] + length * req.dst_aux_index
            transfer_blocks.append(
                (
                    src_addr,
                    dst_addr,
                    length,
                    self.ptr_to_mr_id[prefill_aux_ptrs[i]],
                    dst_aux_ptrs[i],
                    dst_aux_metas[i],
                )
            )

        return self._transfer_data(req.mooncake_session_id, transfer_blocks)

    def _send_mamba_state(
        self,
        req: TransferInfo,
        prefill_mamba_index: list[int],
        dst_state_data_ptrs: list[int],
    ):
        assert len(prefill_mamba_index) == 1, "Mamba should have single state index"
        dst_state_metas = self._get_uccl_metas(req.mooncake_session_id, dst_state_data_ptrs)
        if len(dst_state_metas) != len(dst_state_data_ptrs):
            logger.error("UCCL metadata is missing or mismatched for state transfer")
            return -1

        logger.info(
            "UCCL send_mamba_state: session=%s dst_ptrs=%d",
            req.mooncake_session_id,
            len(dst_state_data_ptrs),
        )
        prefill_state_data_ptrs = self.kv_args.state_data_ptrs
        prefill_state_item_lens = self.kv_args.state_item_lens
        transfer_blocks = []

        for i, dst_state_ptr in enumerate(dst_state_data_ptrs):
            length = prefill_state_item_lens[i]
            src_addr = prefill_state_data_ptrs[i] + length * int(prefill_mamba_index[0])
            dst_addr = dst_state_ptr + length * int(req.dst_state_indices[0])
            transfer_blocks.append(
                (
                    src_addr,
                    dst_addr,
                    length,
                    self.ptr_to_mr_id[prefill_state_data_ptrs[i]],
                    dst_state_ptr,
                    dst_state_metas[i],
                )
            )

        return self._transfer_data(req.mooncake_session_id, transfer_blocks)

    def _send_uccl_endpoint_info(
        self, remote: str, dst_port: int, session_id: str
    ) -> None:
        socket = self._connect(
            format_tcp_address(remote, dst_port), is_ipv6=is_valid_ipv6_address(remote)
        )
        endpoint_metadata = self.engine.get_metadata_for_send()
        meta_preview = endpoint_metadata[:8].hex()
        meta_ip, meta_port, meta_gpu = self.engine.parse_metadata(endpoint_metadata)
        logger.info(
            "Sending UCCL endpoint info to %s:%s for session %s "
            "(metadata_len=%s preview=%s meta_ip=%s meta_port=%s meta_gpu=%s local_ip=%s rank_port=%s)",
            remote,
            dst_port,
            session_id,
            len(endpoint_metadata),
            meta_preview,
            meta_ip,
            meta_port,
            meta_gpu,
            self.local_ip,
            self.rank_port,
        )
        socket.send_multipart(
            [
                UcclKVManager.UCCL_ENDPOINT_HEADER,
                session_id.encode("ascii"),
                self.local_ip.encode("ascii"),
                str(self.rank_port).encode("ascii"),
                endpoint_metadata,
            ]
        )

    def _send_uccl_meta(
        self,
        remote: str,
        dst_port: int,
        session_id: str,
        kv_metas: List[bytes],
        aux_metas: List[bytes],
        state_metas: List[bytes],
    ) -> None:
        socket = self._connect(
            format_tcp_address(remote, dst_port), is_ipv6=is_valid_ipv6_address(remote)
        )
        logger.info(
            "Sending UCCL metadata to %s:%s for session %s (kv=%s aux=%s state=%s)",
            remote,
            dst_port,
            session_id,
            len(kv_metas),
            len(aux_metas),
            len(state_metas),
        )
        socket.send_multipart(
            [
                UcclKVManager.UCCL_META_HEADER,
                session_id.encode("ascii"),
                str(len(kv_metas)).encode("ascii"),
                str(len(aux_metas)).encode("ascii"),
                str(len(state_metas)).encode("ascii"),
                b"".join(kv_metas),
                b"".join(aux_metas),
                b"".join(state_metas),
            ]
        )

    def _handle_uccl_meta(self, msg: List[bytes]):
        session_id = msg[1].decode("ascii")
        logger.info("Received UCCL metadata for session %s", session_id)
        kv_count = int(msg[2].decode("ascii"))
        aux_count = int(msg[3].decode("ascii"))
        state_count = int(msg[4].decode("ascii"))
        kv_blob = msg[5]
        aux_blob = msg[6]
        state_blob = msg[7]

        kv_metas = [
            kv_blob[i * 64 : (i + 1) * 64] for i in range(kv_count)
        ]
        aux_metas = [
            aux_blob[i * 64 : (i + 1) * 64] for i in range(aux_count)
        ]
        state_metas = [
            state_blob[i * 64 : (i + 1) * 64] for i in range(state_count)
        ]

        if session_id not in self.decode_kv_args_table:
            logger.error("UCCL metadata received before registration for %s", session_id)
            return

        info = self.decode_kv_args_table[session_id]
        info.dst_kv_metas = kv_metas
        info.dst_aux_metas = aux_metas
        info.dst_state_metas = state_metas
        info.meta_ready.set()
        logger.info("UCCL meta_ready set for session %s", session_id)

    def _get_uccl_metas(
        self, mooncake_session_id: str, dst_data_ptrs: list[int]
    ) -> Optional[List[bytes]]:
        info = self.decode_kv_args_table.get(mooncake_session_id)
        if info is None:
            logger.error(
                "UCCL metadata lookup failed (missing session %s)",
                mooncake_session_id,
            )
            return None
        if not info.meta_ready.is_set():
            logger.info(
                "UCCL metadata not ready yet for session %s",
                mooncake_session_id,
            )
            return None
        if dst_data_ptrs == info.dst_kv_ptrs:
            return info.dst_kv_metas
        if dst_data_ptrs == info.dst_aux_ptrs:
            return info.dst_aux_metas
        if dst_data_ptrs == info.dst_state_data_ptrs:
            return info.dst_state_metas
        logger.error(
            "UCCL metadata lookup failed (ptrs mismatch) for session %s",
            mooncake_session_id,
        )
        return None

    def _wait_for_uccl_meta(self, info: KVArgsRegisterInfo) -> bool:
        if info.meta_ready.is_set():
            return True
        timeout = get_int_env_var("SGLANG_DISAGGREGATION_UCCL_META_TIMEOUT", 60)
        logger.info(
            "UCCL waiting for meta: session=%s timeout=%s",
            getattr(info, "mooncake_session_id", "unknown"),
            timeout,
        )
        return info.meta_ready.wait(timeout=timeout)

    def _wait_for_uccl_conn(self, info: UcclKVArgsRegisterInfo) -> bool:
        if info.connection_ready.is_set():
            return True
        timeout = get_int_env_var("SGLANG_DISAGGREGATION_UCCL_CONN_TIMEOUT", 60)
        deadline = time.time() + timeout
        logger.info(
            "UCCL waiting for connection: session=%s timeout=%s",
            info.mooncake_session_id,
            timeout,
        )
        while time.time() < deadline:
            if self.engine.get_conn_id(info.mooncake_session_id) is not None:
                info.connection_ready.set()
                logger.info(
                    "UCCL connection_ready set for session %s",
                    info.mooncake_session_id,
                )
                return True
            time.sleep(0.05)
        return False

    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )
                polls = []
                dst_ranks_infos = []
                local_rank = self.attn_tp_rank * self.pp_size + self.pp_rank
                for req in reqs_to_be_processed:
                    if not req.is_dummy:
                        logger.info(
                            "UCCL transfer_worker: room=%s session=%s dst=%s:%s kv_chunk=%s kv_chunk_size=%d",
                            kv_chunk.room,
                            req.mooncake_session_id,
                            req.endpoint,
                            req.dst_port,
                            kv_chunk.index_slice,
                            kv_chunk.prefill_kv_indices.size,
                        )
                        with self.session_lock:
                            if req.mooncake_session_id in self.failed_sessions:
                                self.record_failure(
                                    kv_chunk.room,
                                    f"Decode instance could be dead, remote session {req.mooncake_session_id} is not alive",
                                )
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint,
                                    req.dst_port,
                                    req.room,
                                    KVPoll.Failed,
                                    local_rank,
                                )
                                break

                        chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]
                        if len(chunked_dst_kv_indice) < len(kv_chunk.prefill_kv_indices):
                            kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
                                : len(chunked_dst_kv_indice)
                            ]

                        target_rank_registration_info: UcclKVArgsRegisterInfo = (
                            self.decode_kv_args_table[req.mooncake_session_id]
                        )
                        if not self._wait_for_uccl_conn(target_rank_registration_info):
                            self.record_failure(
                                kv_chunk.room,
                                "Timed out waiting for UCCL connection acceptance.",
                            )
                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_decode_endpoint(
                                req.endpoint,
                                req.dst_port,
                                req.room,
                                KVPoll.Failed,
                                local_rank,
                            )
                            break
                        if not self._wait_for_uccl_meta(target_rank_registration_info):
                            self.record_failure(
                                kv_chunk.room,
                                "Timed out waiting for UCCL metadata from decode instance.",
                            )
                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_decode_endpoint(
                                req.endpoint,
                                req.dst_port,
                                req.room,
                                KVPoll.Failed,
                                local_rank,
                            )
                            break

                        if self.is_mla_backend or (
                            self.attn_tp_size
                            == target_rank_registration_info.dst_attn_tp_size
                        ):
                            logger.info(
                                "UCCL transfer_worker: send_kvcache session=%s",
                                req.mooncake_session_id,
                            )
                            ret = self.send_kvcache(
                                req.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.dst_kv_ptrs,
                                chunked_dst_kv_indice,
                                executor,
                            )
                        else:
                            logger.info(
                                "UCCL transfer_worker: send_kvcache_slice session=%s",
                                req.mooncake_session_id,
                            )
                            ret = self.send_kvcache_slice(
                                req.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.dst_kv_ptrs,
                                chunked_dst_kv_indice,
                                target_rank_registration_info.dst_tp_rank,
                                target_rank_registration_info.dst_attn_tp_size,
                                target_rank_registration_info.dst_kv_item_len,
                                executor,
                            )
                        if ret != 0:
                            with self.session_lock:
                                self.session_failures[req.mooncake_session_id] += 1
                                if self.session_failures[req.mooncake_session_id] >= 1:
                                    self.failed_sessions.add(req.mooncake_session_id)
                                    logger.error(
                                        f"Session {req.mooncake_session_id} failed."
                                    )
                            self.record_failure(
                                kv_chunk.room,
                                f"Failed to send kv chunk of {kv_chunk.room} to {req.endpoint}:{req.dst_port}",
                            )
                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_decode_endpoint(
                                req.endpoint,
                                req.dst_port,
                                req.room,
                                KVPoll.Failed,
                                local_rank,
                            )
                            break

                        if kv_chunk.is_last:
                            if kv_chunk.state_indices is not None:
                                if not self.is_mla_backend and (
                                    self.attn_tp_size
                                    != target_rank_registration_info.dst_attn_tp_size
                                ):
                                    raise RuntimeError(
                                        "PD Disaggregation does NOT support PD different TP sizes for non-MLA hybrid models yet."
                                    )

                                logger.info(
                                    "UCCL transfer_worker: maybe_send_extra session=%s",
                                    req.mooncake_session_id,
                                )
                                self.maybe_send_extra(
                                    req,
                                    kv_chunk.state_indices,
                                    target_rank_registration_info.dst_state_data_ptrs,
                                    executor,
                                )

                            logger.info(
                                "UCCL transfer_worker: send_aux session=%s",
                                req.mooncake_session_id,
                            )
                            ret = self.send_aux(
                                req,
                                kv_chunk.prefill_aux_index,
                                target_rank_registration_info.dst_aux_ptrs,
                            )
                            polls.append(True if ret == 0 else False)
                            dst_ranks_infos.append(
                                (req.endpoint, req.dst_port, req.room)
                            )

                            if len(polls) == req.required_dst_info_num:
                                status = KVPoll.Success if all(polls) else KVPoll.Failed
                                self.update_status(req.room, status)
                                for endpoint, dst_port, room in dst_ranks_infos:
                                    self.sync_status_to_decode_endpoint(
                                        endpoint, dst_port, room, status, local_rank
                                    )
                    else:
                        if kv_chunk.is_last and req.room in self.request_status:
                            self.update_status(req.room, KVPoll.Success)

                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)

            except Exception as e:
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )

    def start_prefill_thread(self):
        def bootstrap_thread():
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                if waiting_req_bytes[0] == UcclKVManager.UCCL_META_HEADER:
                    self._handle_uccl_meta(waiting_req_bytes)
                    continue
                room = waiting_req_bytes[0].decode("ascii")
                mooncake_session_id = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    self.decode_kv_args_table[mooncake_session_id] = (
                        UcclKVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )
                    logger.info(
                        "Registered UCCL decode KVArgs for session %s (dst_port=%s)",
                        mooncake_session_id,
                        waiting_req_bytes[2].decode("ascii"),
                    )
                    with self.session_lock:
                        if mooncake_session_id in self.failed_sessions:
                            self.failed_sessions.remove(mooncake_session_id)
                        if mooncake_session_id in self.session_failures:
                            del self.session_failures[mooncake_session_id]
                    logger.debug(
                        f"Register KVArgs from {mooncake_session_id} successfully"
                    )
                    self._send_uccl_endpoint_info(
                        waiting_req_bytes[1].decode("ascii"),
                        int(waiting_req_bytes[2].decode("ascii")),
                        mooncake_session_id,
                    )
                    continue
                else:
                    required_dst_info_num = int(waiting_req_bytes[7].decode("ascii"))
                    room = int(room)
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}

                    self.transfer_infos[room][mooncake_session_id] = (
                        TransferInfo.from_zmq(waiting_req_bytes)
                    )
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread).start()

    def start_decode_thread(self):
        def decode_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                if msg[0] == UcclKVManager.AUX_DATA_HEADER:
                    self._handle_aux_data(msg)
                    continue
                if msg[0] == UcclKVManager.UCCL_ENDPOINT_HEADER:
                    session_id = msg[1].decode("ascii")
                    prefill_ip = msg[2].decode("ascii")
                    prefill_rank_port = int(msg[3].decode("ascii"))
                    prefill_metadata = msg[4]
                    logger.info(
                        "Received UCCL endpoint info from %s:%s for session %s (metadata_len=%s)",
                        prefill_ip,
                        prefill_rank_port,
                        session_id,
                        len(prefill_metadata),
                    )

                    logger.info(
                        "UCCL decode: connect_endpoint start session=%s prefill_ip=%s",
                        session_id,
                        prefill_ip,
                    )
                    conn_id = self.engine.connect_endpoint(
                        prefill_ip, prefill_metadata
                    )
                    logger.info(
                        "UCCL decode: connect_endpoint result session=%s conn_id=%s",
                        session_id,
                        conn_id,
                    )
                    if conn_id is None:
                        logger.error(
                            "Failed to connect UCCL session for prefill %s",
                            prefill_ip,
                        )
                        continue

                    kv_metas = self.engine.batch_advertise(
                        conn_id,
                        self.kv_args.kv_data_ptrs,
                        self.kv_args.kv_data_lens,
                        self.kv_mr_ids,
                    )
                    aux_metas = self.engine.batch_advertise(
                        conn_id,
                        self.kv_args.aux_data_ptrs,
                        self.kv_args.aux_data_lens,
                        self.aux_mr_ids,
                    )
                    state_metas = self.engine.batch_advertise(
                        conn_id,
                        self.kv_args.state_data_ptrs,
                        self.kv_args.state_data_lens,
                        self.state_mr_ids,
                    )
                    logger.info(
                        "Prepared UCCL metadata for session %s (kv=%s aux=%s state=%s)",
                        session_id,
                        len(kv_metas),
                        len(aux_metas),
                        len(state_metas),
                    )
                    self._send_uccl_meta(
                        prefill_ip,
                        prefill_rank_port,
                        session_id,
                        kv_metas,
                        aux_metas,
                        state_metas,
                    )
                    continue

                (bootstrap_room, status, prefill_rank) = msg
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                prefill_rank = int(prefill_rank.decode("ascii"))

                if status == KVPoll.Success:
                    if bootstrap_room in self.request_status:
                        self.prefill_response_tracker[bootstrap_room].add(prefill_rank)
                        expected_response_num = (
                            self.required_prefill_response_num_table[bootstrap_room]
                        )
                        arrived_response_num = len(
                            self.prefill_response_tracker[bootstrap_room]
                        )
                        if arrived_response_num == expected_response_num:
                            self.update_status(bootstrap_room, KVPoll.Success)
                elif status == KVPoll.Failed:
                    self.record_failure(
                        bootstrap_room,
                        "Failed to get kvcache from prefill instance, it might be dead",
                    )
                    self.update_status(bootstrap_room, status)

        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_dp_size_table.keys())

                for bootstrap_addr in addresses:
                    session = None
                    try:
                        with self.session_pool_lock:
                            session = self.session_pool[bootstrap_addr]
                        response = session.get(
                            f"http://{bootstrap_addr}/health",
                            timeout=(2, 3),
                            headers={"Connection": "keep-alive"},
                        )
                        if response.status_code == 200:
                            self.heartbeat_failures[bootstrap_addr] = 0

                            current_rooms = self.addr_to_rooms_tracker[
                                bootstrap_addr
                            ].copy()

                            for bootstrap_room in current_rooms:
                                if bootstrap_room not in self.request_status:
                                    self.addr_to_rooms_tracker[bootstrap_addr].discard(
                                        bootstrap_room
                                    )
                        else:
                            logger.info(
                                f"Attempting to reconnect to {bootstrap_addr}..."
                            )
                            self.heartbeat_failures[bootstrap_addr] = (
                                self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                            )
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception:
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=decode_thread).start()
        threading.Thread(target=heartbeat_checker).start()


class UcclKVSender(MooncakeKVSender):
    pass


class UcclKVReceiver(MooncakeKVReceiver):
    pass


class UcclKVBootstrapServer(MooncakeKVBootstrapServer):
    pass
