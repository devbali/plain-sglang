from __future__ import annotations

# p95 overheads
DECODE_CONST_PER_SCHEDULING_PASS_OVERHEAD = 0.200
PREFILL_CONST_PER_SCHEDULING_PASS_OVERHEAD = 0.080

# microbenchmark allows this to be anything above 10 ms technically/theoretically
TBT_DELTA = 0.080

# ------

CONST_INTERVAL_DECODE = DECODE_CONST_PER_SCHEDULING_PASS_OVERHEAD / 10
CONST_INTERVAL_PREFILL = PREFILL_CONST_PER_SCHEDULING_PASS_OVERHEAD

PREFILL_PIECEWISE_BOUND_TOTAL_BATCH_SUM = 4064.0

def pooled_decode_time_estimation(
    total_batch_sum: int, max_token_size: int, batch_length: int, n: int
) -> float:
    return CONST_INTERVAL_DECODE + max(
        5e-3,
        1.18629080e-02
        + 6.81862494e-08*total_batch_sum
        + 2.62872519e-07*max_token_size
        + 5.65921863e-05*batch_length
    )


def pooled_prefill_time_estimation_old(
    total_batch_sum: int, max_token_size: int, batch_length: int, n: int
) -> float:
    return CONST_INTERVAL_PREFILL + max(
        5e-3,
        -1.19603713e-02
        + 6.60140608e-05*total_batch_sum 
        + 8.86754786e-06*max_token_size
        + -1.97662090e-04*batch_length
    )


def pooled_prefill_time_estimation(
    total_batch_sum: int, max_token_size: int, batch_length: int, n: int
) -> float:
    del n
    if total_batch_sum <= PREFILL_PIECEWISE_BOUND_TOTAL_BATCH_SUM:
        return CONST_INTERVAL_PREFILL + max(
            5e-3,
            9.15608285e-03
            + 6.14834557e-05 * total_batch_sum
            + 2.26526916e-06 * max_token_size
            + -1.52741501e-05 * batch_length
        )
    return CONST_INTERVAL_PREFILL + max(
        5e-3,
        -4.03734644e-02
        + 6.62669482e-05 * total_batch_sum
        + 1.42083211e-05 * max_token_size
        + -1.02748344e-04 * batch_length
    )


def pooled_cache_prefill_time_estimation(
    total_batch_sum: int, max_token_size: int, batch_length: int, n: int
) -> float:
    return pooled_prefill_time_estimation(total_batch_sum, max_token_size, batch_length, n) - CONST_INTERVAL_PREFILL

# shouldnt be hard coded but need to choose something
MAX_POOLED_DECODE_LATENCY = pooled_decode_time_estimation(328784, 8192, 256, 1)
print(f"MAX_POOLED_DECODE_LATENCY={MAX_POOLED_DECODE_LATENCY}, interval={CONST_INTERVAL_DECODE}, tbt delta = {TBT_DELTA}, total = {CONST_INTERVAL_DECODE +MAX_POOLED_DECODE_LATENCY + TBT_DELTA}")


def isolated_decode_time_estimation(
    total_batch_sum: int, max_token_size: int, batch_length: int, n: int
) -> float:
    return CONST_INTERVAL_DECODE + max(
        MAX_POOLED_DECODE_LATENCY,
        1.18629080e-02
        + 6.81862494e-08*n*total_batch_sum
        + 2.62872519e-07*n*max_token_size
        + 5.65921863e-05*n*batch_length
    ) + TBT_DELTA


def isolated_prefill_time_estimation_old(
    total_batch_sum: int, max_token_size: int, batch_length: int, n: int
) -> float:
    return CONST_INTERVAL_PREFILL + max(
        5e-3,
        -1.19603713e-02
        + 6.60140608e-05*n*total_batch_sum 
        + 8.86754786e-06*n*max_token_size
        + -1.97662090e-04*n*batch_length
    )


def isolated_prefill_time_estimation(
    total_batch_sum: int, max_token_size: int, batch_length: int, n: int
) -> float:
    if total_batch_sum <= PREFILL_PIECEWISE_BOUND_TOTAL_BATCH_SUM:
        return max(
            5e-3,
            9.15608285e-03
            + 6.14834557e-05 * n * total_batch_sum
            + 2.26526916e-06 * n * max_token_size
            + -1.52741501e-05 * n * batch_length
        )
    return max(
        5e-3,
        -4.03734644e-02
        + 6.62669482e-05 * n * total_batch_sum
        + 1.42083211e-05 * n * max_token_size
        + -1.02748344e-04 * n * batch_length
    )
