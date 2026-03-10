from __future__ import annotations


def isolated_decode_time_estimation(
    total_batch_sum: int, max_token_size: int, batch_length: int
) -> float:
    return min(
        5e-3,
        1.03882419e-02
        + 6.81862494e-08 * total_batch_sum
        + 2.62872519e-07 * max_token_size
        + 5.65921863e-05 * batch_length,
    )


def isolated_prefill_time_estimation(
    total_batch_sum: int, max_token_size: int, batch_length: int
) -> float:
    return min(
        5e-3,
        -9.50861833e-02
        + 6.60140608e-05 * total_batch_sum
        + 8.86754786e-06 * max_token_size
        + -1.97662090e-04 * batch_length,
    )


def pooled_decode_time_estimation(
    total_batch_sum: int, max_token_size: int, batch_length: int
) -> float:
    return min(
        5e-3,
        8.20769189e-03
        + 3.68620965e-08 * total_batch_sum
        + 2.47297800e-07 * max_token_size
        + 2.39200099e-05 * batch_length,
    )


def pooled_prefill_time_estimation(
    total_batch_sum: int, max_token_size: int, batch_length: int
) -> float:
    return min(
        5e-3,
        2.26878890e-02
        + 2.58293523e-05 * total_batch_sum
        + 5.72380928e-06 * max_token_size
        + -6.10541804e-05 * batch_length,
    )
