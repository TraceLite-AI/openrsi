# Terminal-Bench five-generation run: verifier-failure audit

**Completed, but invalid for comparing method quality.** This record retracts the earlier interpretation as a valid negative method result. Rewards and decisions are not rewritten. Zero accepted edits is an observed outcome; widespread verifier startup failures prevent an efficacy conclusion about these policies, Terra, or RSI generally.

[中文报告](README.zh-CN.md) · [Recompute](verify.py) · [Summary](audit-summary.json) · [Records](records.json)

## Executed protocol

Trial window: 2026-11-03 21:32:22 to 2026-11-05 04:39:55, Asia/Shanghai.

Three policies selected whole patches, the best single edit, or a merge of positive edits. **No independent native HyperAgents reference arm ran.** The local split was 50 train / 19 gate / 20 sealed, seed 0; identity with unpublished paper splits is not established.

The target was Terra with the pinned MiniSWE seed and documented dependency/API adaptations: high evaluation reasoning, 100 steps, $3 task limit, four concurrent tasks. A fixed direct LLM proposer had **no explicit reasoning-effort setting**; it was not the official Codex/xhigh HyperAgents mutator. Only `mini.yaml` and `agents/default.py` were editable, with at most 12 changed lines per hunk. Each pool froze two two-edit bundles and two hash-selected single edits; merge size was at most two.

The controller used AnyRSI components and RSIHub native scoring, sharing same-parent/same-generation evidence. This was neither three independent seeds nor independently billed equal-token searches. It froze versions before final scoring in 50/19/20 batches, reusing results only for identical final versions.

AnyRSI execution source: `826439524d28406d7c76cdb528fc3ad470fdd487`; RSIHub baseline: `d6554299c58bf0fe2db04ddda0277f9db67ea680`. Recorded adaptations mean this is **not a byte-identical upstream reproduction**. All six recorded controller/runtime pins matched their post-run bytes. See `locked-protocol.json` inside `records.json`.

## Verifier startup failures became ordinary zeros

All **1,061 physical task trials** were audited:

| Phase | Read-only Python-directory failure | Trials |
|---|---:|---:|
| Search-time training | 110 | 250 |
| Search-time gate/candidates/confirmation | 493 | 722 |
| Post-freeze final evaluation | 40 | 89 |
| Total | **643** | **1,061** |

All 643 logs contained `Read-only file system` and `/opt/evolve/uv/python/`, without reaching pytest. **642 were recorded as `benchmark_complete`, reward zero.** One additionally had an Agent timeout, so these categories overlap. Final evaluation was affected on 22/50 train, 13/19 gate and 5/20 sealed tasks; two additional final tasks had API-policy refusals. Other zeros have not all been certified as genuine task failures.

The evidence chain:

1. Verifiers inherited `UV_PYTHON_INSTALL_DIR=/opt/evolve/uv/python`, mounted read-only; the shared directory contained Python 3.12.
2. For example, the original `mteb-retrieve` verifier requires `uvx -p 3.13`.
3. Interpreter installation failed before pytest; the shell script still wrote zero for the nonzero command exit.
4. The runner usually recorded no exception. The experiment's coverage check treated these zeros as scoreable.
5. The proposer received zero rewards and Agent trace tails, not the verifier bootstrap error, and proposed behavior reminders for environment failures.

See [per-trial audit](evidence/verifier-audit.json) and [bootstrap evidence](evidence/bootstrap.json). **Do not remove these cases and advertise an adjusted benchmark score.** Missing valid assessments require a new experiment. Earlier hourly reports checked runner exceptions rather than whether the verifier reached its tests; that health check was insufficient.

## Raw outcomes, not an efficacy result

Every policy accepted zero modifications and ended at version `0`.

| Generation | Parent initial/confirmation | Whole | Single | Nominee confirmation | Accepted |
|---|---|---|---|---|---:|
| 1 | 4/5 | 5, 5 | 5, 4 | whole 5; single 5 | 0 |
| 2 | 4/5 | 5, 5 | 5, 5 | whole 4; single 5; merged 5 | 0 |
| 3 | 4/5 | 4, 4 | 4, 3 | no positive nominee | 0 |
| 4 | 4/5 | 5, 5 | 5, 5 | whole 5; single 5; merged 5 | 0 |
| 5 | 5/4 | 4, 5 | 5, 5 | no positive nominee | 0 |

Each denominator is 19 scheduled tasks, not 19 assessments newly certified as valid. Final raw counts: train **22/50**, gate **4/19**, sealed **9/20**, total **35/89**. Since final versions were identical, all policies reference one final evaluation, not three matching independent replications.

## Secondary observations and limits

**Edits loaded:** all 22 candidates changed only `mini.yaml`'s `instance_template`. No Python functionality changed. In 532 candidate traces, the template matched the tagged version and the rendered first user message contained all added text. An edit-loading failure is not supported. [Activation evidence](evidence/prompt-activation.json)

**Generic proposals:** edits mostly asked for format, invocation, source and artifact checks. Across five pools, 32 atomic proposals were retained, 20 entered bundles and only 10 were individually tested. Broader edits might matter, but this invalid evaluator cannot establish that they would help. [Actual diffs](patches/)

**Few varying tasks:** in ten parent gate evaluations, `gcode-to-text` passed 7 times and `portfolio-optimization` 8 times; across 20 initial candidates, 15 and 18 times. The other 17 parent outcomes were fixed. No initial candidate rescued a consistently failing gate task, but 13 of the 14 such tasks repeatedly failed verifier startup. Persistent zero cannot simply be interpreted as persistent agent inability.

**No accumulated search experience:** all policies retained version 0. Each generation reset its proposal pool; `already_kept` referred to the current generation only. Rejected-edit history, momentum, improver evolution and weight training were not enabled. This describes the executed controller, not AnyRSI's full capabilities.

**Unconfirmed gains were filtered:** most one-task initial gains disappeared when the unchanged parent was repeated. The records do not prove that strict acceptance rejected true improvements. Tie acceptance requires a separate valid ablation.

## Resources

- 1,061 trials: 250 train, 722 gate, 89 final.
- MiniSWE recorded 18,267 model calls; not necessarily every internal LiteLLM HTTP retry.
- Estimated Target cost **$98.45719345**, including **$65.806426** on trials with the read-only failure. Not a provider bill; host costs excluded.
- Proposer: **16 HTTP attempts**, one without usage, **254,469 reported tokens**. Its HTTP count is not added to the Target logical-call count as a single HTTP metric.
- Credit-exhaustion cohorts, single-step pilots, symbolic tasks and FlashRSI are excluded.

## Repair measurement first

Separate Agent and verifier interpreter environments; validate the task-required interpreter or use a writable isolated verifier install directory. Do not broadly make host-shared directories writable. Exercise real verifier bootstrap with known-good and known-bad artifacts before paid trials. Agent installation smoke alone is insufficient.

Classify recognized setup failures as unscoreable and stop on inadequate coverage. Start a new protocol/runtime identity and result directory after repair. Preserve historical rewards and decisions; do not splice selectively rerun failures into an improved score. Only then test mechanism-specific edits, richer evidence, diversity and rejected-edit history separately.

This work is a read-only audit and publication. **No new paid experiment was started.**

## Verify and inspect

```bash
python benchmarks/terminal_multigen_20261103/verify.py
```

Expected: `integrity: passed` and `valid_for_method_comparison: false`. The first means the public exports reconcile, not that the experiment is valid.

- `records.json`: sanitized protocol, all 46 native receipts, lineage, pools and original hashes.
- `patches/`: 22 full candidate diffs.
- `controller/`: path-redacted copies of executed code and original hashes; placeholder copies are not turnkey scripts.
- `evidence/`: full verifier audit, activation checks, timings, partition identities and environment evidence.
- `MANIFEST.json`: exported-file hashes, distinct from original-file hashes.
- Full tasks, answers, API credentials, host addresses, unredacted configuration and complete trajectories remain in the original run store. Verification does not call models or establish server-side weight identity.
- MiniSWE patch context is covered by [MINI_SWE_LICENSE.txt](MINI_SWE_LICENSE.txt).
