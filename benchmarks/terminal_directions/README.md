# Fixed-parent Terminal-Bench pilot

Freeze one MiniSWE source tree, one evaluator, seed 0 and the 50/19/20 task partition before model calls. Generate proposals using only the first five round-one **train** tasks. Do not mix these results with a previous 30-task run.

1. Measure the unchanged parent three times on all 19 selection tasks. Keep each trial, task reward, exception type, model calls, tokens and runtime receipt. Count flips only among tasks with a clean result on all three repeats. Preregistered pilot stop: fewer than 16 clean tasks or more than four clean flips.
2. Freeze two distinct bounded-hunk proposals from the same parent and train evidence, within eight proposer HTTP attempts. Require at least one multi-op proposal. Before looking at selection outcomes, select two distinct atoms by content hash. If the pool is degenerate, stop and report it.
3. Measure all four candidates on the same 19 tasks (reuse exactly identical versions). All policies share this panel. A selects the best positive whole proposal, B the best positive atom, C merges up to two positive atoms. Use the mean parent reward per task over the three repeats and the common clean cases for all policies. Fewer than 16 common clean cases stops the comparison.
4. Re-evaluate the parent and each unique nominee: a positive paired gain is required for acceptance. Freeze the resulting snapshots **before** using the 20 sealed tasks. Report native scores plus paired common-clean differences; keep exceptions separate. Each physical batch runs at most four trials concurrently.

This compares choices on a common measurement table. Charge the panel once to the experiment and report each arm's useful measurements separately. It is **not** three independently billed processes with identical token use. Charge train evidence, proposal attempts, selection, confirmation and final separately. A positive result in this one-parent pilot is not evidence of a stable multi-generation improvement.
