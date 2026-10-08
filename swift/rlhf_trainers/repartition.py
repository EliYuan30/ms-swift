# Copyright (c) ModelScope Contributors. All rights reserved.
"""Point-to-point re-partition of turn-split rollout outputs.

Turn splitting changes how many training samples each rank holds, so the trainer re-balances them:
the reference path all-gathers every rank's rollout outputs to every rank, sorts them by request id
and keeps one contiguous even slice. For video rollouts (frames carried as data URIs) every rank then
receives the whole step's payload. Here ranks gather only request ids, derive the identical
assignment, and send each output once to its destination with a single byte ``all_to_all``.
"""
import pickle
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


def even_slices(total: int, num_procs: int) -> List[Tuple[int, int]]:
    """Contiguous slices identical to ``get_even_process_data``."""
    base, remainder = divmod(total, num_procs)
    slices = []
    for rank in range(num_procs):
        if rank < remainder:
            start = rank * (base + 1)
            end = start + base + 1
        else:
            start = remainder * (base + 1) + (rank - remainder) * base
            end = start + base
        slices.append((start, end))
    return slices


def plan_repartition(request_ids_by_rank: Sequence[Sequence[str]], num_procs: int) -> List[List[Tuple[int, int]]]:
    """Per destination rank, the ``(source_rank, local_index)`` items in reference order.

    The reference flattens outputs in rank order, sorts stably by request id and splits evenly.
    """
    flat = [(request_id, source, index) for source, request_ids in enumerate(request_ids_by_rank)
            for index, request_id in enumerate(request_ids)]
    flat.sort(key=lambda item: item[0])
    return [[(source, index) for _, source, index in flat[start:end]]
            for start, end in even_slices(len(flat), num_procs)]


def all_to_all_objects(send_lists: Sequence[list]) -> List[list]:
    """``received[source]`` is the list ``source`` sent to this rank (one pickled byte all_to_all)."""
    payloads = [pickle.dumps(list(items), protocol=pickle.HIGHEST_PROTOCOL) for items in send_lists]
    backend = dist.get_backend()
    device = (torch.device('cuda', torch.cuda.current_device())
              if backend == 'nccl' and torch.cuda.is_available() else torch.device('cpu'))
    send_sizes = torch.tensor([len(payload) for payload in payloads], dtype=torch.long, device=device)
    recv_sizes = torch.empty_like(send_sizes)
    dist.all_to_all_single(recv_sizes, send_sizes)
    send_buffer = torch.frombuffer(bytearray(b''.join(payloads)), dtype=torch.uint8).to(device)
    recv_split = recv_sizes.tolist()
    recv_buffer = torch.empty(sum(recv_split), dtype=torch.uint8, device=device)
    dist.all_to_all_single(recv_buffer, send_buffer, output_split_sizes=recv_split,
                           input_split_sizes=send_sizes.tolist())
    data = recv_buffer.cpu().numpy().tobytes()
    received, offset = [], 0
    for size in recv_split:
        received.append(pickle.loads(data[offset:offset + size]))
        offset += size
    return received


def repartition_rollout_outputs(outputs: list,
                                rank: int,
                                num_procs: int,
                                gather_fn: Callable[[list], list],
                                exchange_fn: Optional[Callable[[Sequence[list]], List[list]]] = None):
    """Return ``(local_outputs, pad_count)`` equal to sort-by-request-id + even split of the global list."""
    request_ids_by_rank = gather_fn([[output.response.id for output in outputs]])
    plan = plan_repartition(request_ids_by_rank, num_procs)
    total = sum(len(ids) for ids in request_ids_by_rank)
    remainder = total % num_procs
    pad_count = 1 if remainder > 0 and rank >= remainder else 0
    send = [[outputs[index] for source, index in plan[destination] if source == rank]
            for destination in range(num_procs)]
    received = (exchange_fn or all_to_all_objects)(send)
    cursors = [0] * num_procs
    local = []
    for source, _ in plan[rank]:
        local.append(received[source][cursors[source]])
        cursors[source] += 1
    return local, pad_count
