module TauFairScheduler

open FStar.Seq


type perf_coeffs = {
    cp_ns: Prims.int;
    alpha_p_nspt: Prims.int;
    beta_p_nspt: Prims.int;
    gamma_p_nspt: Prims.int;
    cd_ns: Prims.int;
    alpha_d_nspt: Prims.int;
    beta_d_nspt: Prims.int;
    gamma_d_nspt: Prims.int;
}

type tau_breakdown = {
    tau_us: Prims.int;
    tau_cache_us: Prims.int;
    tau_prefill_us: Prims.int;
    tau_decode_us: Prims.int;
}

type token_deadline = {
    rid: Prims.string;
    uid: Prims.string;
    token_index: Prims.int;
    iso_latency_us: Prims.int;
    tau_applied_us: Prims.int;
    deadline_us: Prims.int;
}

type alt_req = {
    rid: Prims.string;
    uid: Prims.string;
    prompt_len: Prims.int;
    completion_len: Prims.int;
    status: Prims.int;
    decode_count: Prims.int;
    arrival_time_us: Prims.int;
    iso_prefill_time_us: Prims.int;
    iso_decode_time_us: Prims.int;
    ant_type: Prims.int;
    ant_end_ts_us: Prims.int;
}


type user_status = {
    uid: Prims.string;
    requests: FStar.Seq.seq alt_req;
    n_requests: Prims.int;
}


type alt_history_state = {
    current_time_us: Prims.int;
    num_users: Prims.int;
    max_kv: Prims.int;
    users: FStar.Seq.seq user_status;
    n_users: Prims.int;
    deadlines: FStar.Seq.seq token_deadline;
    n_deadlines: Prims.int;
}


type sched_req = {
    rid: Prims.string;
    uid: Prims.string;
    prompt_len: Prims.int;
    max_new_tok: Prims.int;
    deadline_us: Prims.int;
    is_running: Prims.bool;
    is_finished: Prims.bool;
}

type sched_batch = {
    reqs: FStar.Seq.seq sched_req;
    sum_tok: Prims.int;
    max_tok: Prims.int;
    req_count: Prims.int;
    is_prefill: Prims.bool;
    is_empty: Prims.bool;
}

type fair_share_state = {
    uid: Prims.string;
    total_kv: Prims.int;
    evictable_kv: Prims.int;
    fair_share: Prims.int;
    reservation: Prims.int;
    pending_prefills: Prims.int;
    running_decodes: Prims.int;
    last_decode_ts_us: Prims.int;
    last_prefill_ts_us: Prims.int;
}

type e_d_f_ordering_key = {
    tier: Prims.int;
    neg_shortfall: Prims.int;
    excess: Prims.int;
}

open FStar.Seq

(* ── krml-compatible helpers ────────────────────────────────────────────── *)

(* Replace multiplication with repeated addition so krml can compile this. *)
let rec mul (a: Prims.int) (b: Prims.int) (acc: Prims.int) : Prims.int =
  if b <= 0 then acc else mul a (b - 1) (acc + a)

(* All inner let-rec have been lifted to top-level, ordered callee-before-caller. *)

(* — Helper for would_violate_decode_deadlines — *)
let rec would_violate_aux
  (running_deadlines_us: seq Prims.int)
  (threshold: Prims.int)
  (i: Prims.int)
  (n_running: Prims.int)
  : Prims.bool
=
  if i >= n_running then false
  else if index running_deadlines_us i <= threshold then true
  else would_violate_aux running_deadlines_us threshold (i + 1) n_running

(* — Helper: gen_range for sort_prefill_queue — *)
let rec gen_range (n: Prims.int) (i: Prims.int) : (seq Prims.int) =
  if i >= n then empty
  else cons i (gen_range n (i + 1))

(* — Helper: remove_elem for sort_prefill_queue — *)
let rec remove_elem (s: seq Prims.int) (v: Prims.int) : (seq Prims.int) =
  if length s = 0 then empty
  else
    let hd = head s in
    let tl = tail s in
    if hd = v then tl
    else cons hd (remove_elem tl v)

(* — Helper: aux for warn_overdue_deadlines — *)
let rec warn_overdue_aux
  (deadlines_us: seq Prims.int)
  (n: Prims.int)
  (now_us: Prims.int)
  (i: Prims.int)
  : Prims.int =
  if i >= n then 0
  else (if index deadlines_us i <= now_us then 1 else 0) + warn_overdue_aux deadlines_us n now_us (i + 1)

(* — Helper: aux for count_prefill_admitted — *)
let rec count_prefill_aux
  (total_kvs: seq Prims.int)
  (fair_shares: seq Prims.int)
  (n: Prims.int)
  (i: Prims.int)
  : Prims.int =
  if i >= n then 0
  else (if index total_kvs i <= index fair_shares i then 1 else 0) + count_prefill_aux total_kvs fair_shares n (i + 1)

(* — Helper: check_decode_deadlines_aux — *)
let rec check_decode_deadlines_aux
  (deadlines_us: seq Prims.int)
  (n: Prims.int)
  (now_us: Prims.int)
  (i: Prims.int)
  (acc: seq Prims.int)
  : seq Prims.int
  =
  if i >= n then acc
  else
    let flag = if index deadlines_us i <= now_us then 1 else 0 in
    check_decode_deadlines_aux deadlines_us n now_us (i + 1) (upd acc i flag)

(* — Helper: sum_aux for compute_global_slack — *)
let rec sum_aux (total_kvs: seq Prims.int) (n: Prims.int) (i: Prims.int) : Prims.int =
  if i >= n then 0
  else index total_kvs i + sum_aux total_kvs n (i + 1)

(* ── Alternate-history deadline model ────────────────────────────────── *)

(* ---- Implementations ---- *)

(* 1 *)
let edf_fairshare_key
  (deadline_us: Prims.int)
  (total_kv: Prims.int)
  (fair_share: Prims.int)
  (tau_us: Prims.int)
  (now_us: Prims.int)
  : e_d_f_ordering_key
=
  let tier =
    if deadline_us <= now_us then 0
    else if deadline_us - now_us <= tau_us then 1
    else 2
  in
  let shortfall = if fair_share > total_kv then fair_share - total_kv else 0 in
  let excess    = if total_kv > fair_share then total_kv - fair_share else 0 in
  { tier; neg_shortfall = -shortfall; excess }

(* 2 *)
let edf_compare (a: e_d_f_ordering_key) (b: e_d_f_ordering_key) : Prims.bool =
  if a.tier <> b.tier then a.tier < b.tier
  else if a.neg_shortfall <> b.neg_shortfall then a.neg_shortfall < b.neg_shortfall
  else a.excess < b.excess

(* 3 *)
let is_well_behaved (total_kv: Prims.int) (fair_share: Prims.int) : Prims.bool =
  total_kv <= fair_share

(* 4 *)
let is_over_quota (total_kv: Prims.int) (fair_share: Prims.int) : Prims.bool =
  total_kv > fair_share

(* 5 *)
let is_within_reservation (total_kv: Prims.int) (reservation: Prims.int) : Prims.bool =
  total_kv <= reservation

(* 6 *)
let unevictable_tokens (total_kv: Prims.int) (evictable_kv: Prims.int) : Prims.int =
  total_kv - evictable_kv

(* 7 *)
let eviction_spare (total_kv: Prims.int) (reservation: Prims.int) : Prims.int =
  if total_kv > reservation then total_kv - reservation else 0

(* 8 *)
let on_new_request_init_deadline
  (alt_state: alt_history_state)
  (uid: Prims.string)
  (prompt_len: Prims.int)
  (completion_len: Prims.int)
  (iso_prefill_time_us: Prims.int)
  (tau_cache_us: Prims.int)
  (tau_prefill_us: Prims.int)
  (now_us: Prims.int)
  : Prims.int
=
  now_us + iso_prefill_time_us + tau_cache_us + tau_prefill_us

(* 17 — defined before prefill_vs_decode_decision since 9 calls it *)
let would_violate_decode_deadlines
  (alt_state: alt_history_state)
  (prefill_cost_us: Prims.int)
  (headroom_us: Prims.int)
  (running_deadlines_us: seq Prims.int)
  (n_running: Prims.int)
  (delta_decode_mt_us: Prims.int)
  (now_us: Prims.int)
  : Prims.bool
=
  let threshold = now_us + prefill_cost_us + delta_decode_mt_us in
  would_violate_aux running_deadlines_us threshold 0 n_running

(* 9 *)
let prefill_vs_decode_decision
  (alt_state: alt_history_state)
  (running_deadlines_us: seq Prims.int)
  (n_running: Prims.int)
  (has_candidate_prefill: Prims.bool)
  (prefill_cost_us: Prims.int)
  (accumulated_headroom_us: Prims.int)
  (delta_decode_mt_us: Prims.int)
  (tau_us: Prims.int)
  (now_us: Prims.int)
  (any_over_quota: Prims.bool)
  (any_starved: Prims.bool)
  (max_per_user_active: Prims.bool)
  : Prims.int
=
  if not has_candidate_prefill then
    (if n_running > 0 then -1 else 0)
  else if prefill_cost_us <= accumulated_headroom_us
       && not any_over_quota
       && not max_per_user_active
       && not (would_violate_decode_deadlines alt_state prefill_cost_us
                 accumulated_headroom_us running_deadlines_us n_running
                 delta_decode_mt_us now_us)
  then 1
  else if n_running > 0 then -1
  else 0

(* — Helper: key_of for sort_prefill_queue — *)
let key_of
  (deadlines_us: seq Prims.int)
  (total_kvs: seq Prims.int)
  (fair_shares: seq Prims.int)
  (tau_us: Prims.int)
  (now_us: Prims.int)
  (i: Prims.int)
  : e_d_f_ordering_key
=
  edf_fairshare_key (index deadlines_us i) (index total_kvs i) (index fair_shares i) tau_us now_us

(* — Helper: find_min for sort_prefill_queue — *)
let rec find_min
  (deadlines_us: seq Prims.int)
  (total_kvs: seq Prims.int)
  (fair_shares: seq Prims.int)
  (tau_us: Prims.int)
  (now_us: Prims.int)
  (s: seq Prims.int)
  : Prims.int
=
  if length s = 1 then head s
  else
    let hd = head s in
    let tl = tail s in
    let rest = find_min deadlines_us total_kvs fair_shares tau_us now_us tl in
    if edf_compare (key_of deadlines_us total_kvs fair_shares tau_us now_us hd)
                   (key_of deadlines_us total_kvs fair_shares tau_us now_us rest)
    then hd else rest

(* — Helper: sort_aux for sort_prefill_queue — *)
let rec sort_aux
  (deadlines_us: seq Prims.int)
  (total_kvs: seq Prims.int)
  (fair_shares: seq Prims.int)
  (tau_us: Prims.int)
  (now_us: Prims.int)
  (s: seq Prims.int)
  (acc: seq Prims.int)
  : (seq Prims.int)=
  if length s = 0 then acc
  else
    let m = find_min deadlines_us total_kvs fair_shares tau_us now_us s in
    sort_aux deadlines_us total_kvs fair_shares tau_us now_us
             (remove_elem s m) (snoc acc m)

(* 10 — selection sort by edf_compare priority *)
let sort_prefill_queue
  (alt_state: alt_history_state)
  (deadlines_us: seq Prims.int)
  (uids: seq Prims.string)
  (total_kvs: seq Prims.int)
  (fair_shares: seq Prims.int)
  (n: Prims.int)
  (tau_us: Prims.int)
  (now_us: Prims.int)
  : seq Prims.int
=
  sort_aux deadlines_us total_kvs fair_shares tau_us now_us (gen_range n 0) empty

(* 11 *)
let warn_overdue_deadlines
  (alt_state: alt_history_state)
  (deadlines_us: seq Prims.int)
  (n: Prims.int)
  (now_us: Prims.int)
  : Prims.int
=
  warn_overdue_aux deadlines_us n now_us 0

(* 12 *)
let count_prefill_admitted
  (alt_state: alt_history_state)
  (uids: seq Prims.string)
  (total_kvs: seq Prims.int)
  (fair_shares: seq Prims.int)
  (n: Prims.int)
  : Prims.int
=
  count_prefill_aux total_kvs fair_shares n 0

(* 13 *)
let check_decode_deadlines
  (alt_state: alt_history_state)
  (deadlines_us: seq Prims.int)
  (n: Prims.int)
  (now_us: Prims.int)
  : seq Prims.int
=
  check_decode_deadlines_aux deadlines_us n now_us 0 (create n 0)

(* 14 *)
let update_headroom
  (alt_state: alt_history_state)
  (old_headroom_us: Prims.int)
  (was_prefill: Prims.bool)
  (prefill_cost_us: Prims.int)
  (was_decode: Prims.bool)
  (delta_decode_iso_us: Prims.int)
  (delta_decode_mt_us: Prims.int)
  (was_idle: Prims.bool)
  : Prims.int
=
  let h0 = old_headroom_us in
  let h1 = if was_prefill then h0 - prefill_cost_us else h0 in
  let h2 = if was_decode then h1 + delta_decode_iso_us - delta_decode_mt_us else h1 in
  let h3 = if was_idle then h2 + delta_decode_iso_us else h2 in
  if h3 >= 0 then h3 else 0

(* 15 *)
let can_admit_request
  (alt_state: alt_history_state)
  (uid: Prims.string)
  (needed_tokens: Prims.int)
  (total_kv: Prims.int)
  (evictable_kv: Prims.int)
  (fair_share: Prims.int)
  (reservation: Prims.int)
  (global_slack: Prims.int)
  (cache_has_space: Prims.bool)
  : Prims.bool
=
  cache_has_space
  || (total_kv + needed_tokens <= fair_share)
  || (total_kv + needed_tokens <= reservation && evictable_kv >= needed_tokens)
  || global_slack >= needed_tokens

(* 16 *)
let compute_global_slack
  (alt_state: alt_history_state)
  (total_kvs: seq Prims.int)
  (n: Prims.int)
  (max_kv: Prims.int)
  : Prims.int
=
  let sum_kv = sum_aux total_kvs n 0 in
  if max_kv >= sum_kv then max_kv - sum_kv else 0

(* 17 *)
let isolated_prefill_latency
  (prompt_len: Prims.int)
  (max_tok: Prims.int)
  (req_count: Prims.int)
  (coeffs: perf_coeffs)
  (num_clients: Prims.int)
  : Prims.int
=
  let term1 = mul coeffs.alpha_p_nspt prompt_len 0 in
  let term2 = mul coeffs.beta_p_nspt max_tok 0 in
  let term3 = mul coeffs.gamma_p_nspt req_count 0 in
  let inner = coeffs.cp_ns + term1 + term2 + term3 in
  let prod = mul num_clients inner 0 in
  prod / 1000

(* 18 *)
let isolated_decode_latency
  (sum_tok: Prims.int)
  (max_tok: Prims.int)
  (req_count: Prims.int)
  (coeffs: perf_coeffs)
  (num_clients: Prims.int)
  : Prims.int
=
  let term1 = mul coeffs.alpha_d_nspt sum_tok 0 in
  let term2 = mul coeffs.beta_d_nspt max_tok 0 in
  let term3 = mul coeffs.gamma_d_nspt req_count 0 in
  let inner = coeffs.cd_ns + term1 + term2 + term3 in
  let prod = mul num_clients inner 0 in
  prod / 1000

(* 19 *)
let mt_decode_latency
  (sum_tok: Prims.int)
  (max_tok: Prims.int)
  (req_count: Prims.int)
  (coeffs: perf_coeffs)
  : Prims.int
=
  let term1 = mul coeffs.alpha_d_nspt sum_tok 0 in
  let term2 = mul coeffs.beta_d_nspt max_tok 0 in
  let term3 = mul coeffs.gamma_d_nspt req_count 0 in
  (coeffs.cd_ns + term1 + term2 + term3) / 1000

(* 20 *)
let compute_headroom (delta_decode_iso_us: Prims.int) (delta_decode_mt_us: Prims.int) : Prims.int =
  delta_decode_iso_us - delta_decode_mt_us

(* 21 *)
let split_tau (tau_us: Prims.int) (cache_ratio_bps: Prims.int) (tau_decode_fixed_us: Prims.int) : tau_breakdown =
  let tau_prod = mul tau_us cache_ratio_bps 0 in
  let tau_cache = tau_prod / 10000 in
  {
    tau_us = tau_us;
    tau_cache_us = tau_cache;
    tau_prefill_us = tau_us - tau_cache;
    tau_decode_us = tau_decode_fixed_us;
  }

(* 22 *)
let compute_new_req_ant_type (req: alt_req) : Prims.int = 0

(* 23 *)
let compute_prefill_done_ant_type (req: alt_req) : Prims.int = 1

(* 24 *)
let compute_decode_done_ant_type (req: alt_req) : Prims.int =
  if req.decode_count < req.completion_len then 1 else -1

(* 25 *)
let compute_ant_prefill_end_us (arrival_time_us: Prims.int) (iso_prefill_time_us: Prims.int) : Prims.int =
  arrival_time_us + iso_prefill_time_us

(* 26 *)
let compute_ant_decode_end_us (previous_ant_end_us: Prims.int) (iso_decode_time_us: Prims.int) : Prims.int =
  previous_ant_end_us + iso_decode_time_us

(* 27 *)
let compute_deadline_from_ant (ant_end_ts_us: Prims.int) (tau_applied_us: Prims.int) : Prims.int =
  ant_end_ts_us + tau_applied_us

(* 28 *)
let alt_history_init (num_users: Prims.int) (start_time_us: Prims.int) (max_kv: Prims.int) : alt_history_state =
  {
    current_time_us = start_time_us;
    num_users = num_users;
    max_kv = max_kv;
    users = Seq.empty;
    n_users = 0;
    deadlines = Seq.empty;
    n_deadlines = 0;
  }

(* 29 *)
let alt_history_push_request
  (state: alt_history_state)
  (rid: Prims.string)
  (uid: Prims.string)
  (prompt_len: Prims.int)
  (completion_len: Prims.int)
  (arrival_time_us: Prims.int)
  (coeffs: perf_coeffs)
  (tb: tau_breakdown)
  : alt_history_state
=
  let iso_prefill = isolated_prefill_latency prompt_len 0 1 coeffs 1 in
  let iso_decode = isolated_decode_latency 1 0 1 coeffs 1 in
  let ant_end = compute_ant_prefill_end_us arrival_time_us iso_prefill in
  let deadline_us = compute_deadline_from_ant ant_end tb.tau_prefill_us in
  let new_req = {
    rid = rid;
    uid = uid;
    prompt_len = prompt_len;
    completion_len = completion_len;
    status = 0;
    decode_count = 0;
    arrival_time_us = arrival_time_us;
    iso_prefill_time_us = iso_prefill;
    iso_decode_time_us = iso_decode;
    ant_type = 0;
    ant_end_ts_us = ant_end;
  } in
  let deadline = {
    rid = rid;
    uid = uid;
    token_index = 0;
    iso_latency_us = iso_prefill;
    tau_applied_us = tb.tau_prefill_us;
    deadline_us = deadline_us;
  } in
  let new_user = { uid = uid; requests = Seq.cons new_req Seq.empty; n_requests = 1 } in
  {
    state with
    users = Seq.snoc state.users new_user;
    n_users = state.n_users + 1;
    deadlines = Seq.snoc state.deadlines deadline;
    n_deadlines = state.n_deadlines + 1;
  }

(* 30 *)
let alt_history_advance (state: alt_history_state) (coeffs: perf_coeffs) (tb: tau_breakdown) : alt_history_state =
  { state with current_time_us = state.current_time_us + 1 }

(* 31 *)
let alt_history_extract_deadlines (state: alt_history_state) : FStar.Seq.seq token_deadline =
  state.deadlines

(* 32 *)
let alt_history_complete_request (state: alt_history_state) (rid: Prims.string) : alt_history_state =
  state

(* 33 *)
let alt_history_reconcile (state: alt_history_state) (real_time_us: Prims.int) (completed_rids: FStar.Seq.seq Prims.string) (n_completed: Prims.int) : alt_history_state =
  { state with current_time_us = real_time_us }

(* 34 *)
let alt_history_get_user_token_count (state: alt_history_state) (uid: Prims.string) : Prims.int =
  0
