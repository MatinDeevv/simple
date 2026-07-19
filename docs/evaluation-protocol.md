# Evaluation Protocol: Causal Intervals, Purging, and Entry Semantics

`engine/evaluation/evaluation_protocol.py` and
`engine/evaluation/entry_diagnostics.py` are shared causal-research helpers.
They do not model execution or PnL. Their interval convention is inclusive:
an observation at `source_index` with horizon `h` occupies
`[source_index, source_index + h]`.

## Policy-sensitive populations

Independent entries remain the frozen primary population. Every secondary
policy refits its comparator from policy-accepted frozen-training rows and
evaluates only policy-accepted OOS rows. Boundary-crossing labels are excluded
from both populations. Results expose both reset-at-OOS and chronological
carry-state contracts; carry state may observe a boundary row's chronology but
never its label for comparator fitting. Neither policy is selected by
performance.
Neutral outcomes remain in the primary three-class target, and these helpers
make no execution or profitability claim.

## Row-specific intervals and segment-scoped purge

`target_interval`, `assign_target_clusters`, and `build_evaluation_metadata`
accept either one non-negative integer horizon or a one-dimensional horizon
array matching `source_index`. Booleans, non-finite values, fractional floats,
negative values, wrong shapes, and mismatched lengths are rejected. Ordering is
stable after source-index sorting and values are aligned back to the caller's
original rows.

`build_evaluation_metadata` reports target starts/ends, target cluster IDs,
plain purge status, and embargo status. A caller may supply a distinct
evaluation mask: rows crossing a chronological split can therefore be excluded
from frozen training without being treated as OOS scoring rows.

Purge only applies where inclusive intervals overlap **and** segment IDs match.
Embargo widens held-out windows numerically but retains their segment identity;
no numerical adjacency across a data gap becomes an overlap. Cluster IDs also
split at every segment transition.

## Entry policy semantics

The frozen default, `independent_research_entries`, accepts every eligible row.
These explicit sensitivity policies are available in deterministic order:

1. `independent_research_entries`;
2. `first_entry_per_signal_episode`;
3. `non_overlapping_global`;
4. `non_overlapping_component`;
5. `non_overlapping_basket`;
6. `one_representative_per_target_cluster` when target clusters are available.

Global, component, and basket non-overlap are scoped within one causal segment;
an open entry in one segment cannot block an entry in the next. The basket key
is component plus a six-decimal diagnostic-weight signature.

`annotate_entry_diagnostics` never mutates its input. It adds an episode ID,
accepted flag, active target counts, component/basket overlap counts, overlap
minutes, candidate turnover, and accepted-entry turnover. It performs a stable
source-order pass while preserving the original row alignment. At a segment
transition it clears open entries and all prior turnover state:

- first candidate after a gap: `candidate_weight_change_l1 = 0`;
- first accepted entry after a gap: `accepted_entry_weight_change_l1 = null`.

This avoids comparing a post-gap basket or target with a pre-gap one. An
explicit insufficient-sample result is required whenever a chosen sensitivity
cannot support the requested uncertainty procedure.
