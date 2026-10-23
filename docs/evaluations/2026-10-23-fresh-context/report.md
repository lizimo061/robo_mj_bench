# Fresh context robotics evaluation blocked by subscription inference

The requested model-control evaluation could not begin: ChatGPT login succeeded,
but two minimal inference checks returned no model response before their deadlines.
No robot server or episode was launched. All eleven families remain unattempted;
these results provide no evidence about the model's robotics ability or task difficulty.

The evaluation started on 2026-10-23; this report was finalized on 2026-10-24
(Asia/Shanghai). The original date remains in the directory name.

## Coverage and denominators

The predeclared budget was one episode per family for coverage, expanding to five
per family (55 total) if inference and execution remained functional. See
[the controller contract](contract.md) and [machine readable results](results.json).

| Task | Started | Completed | Valid | Successful | Success rate | Infrastructure invalid episodes | Status |
|---|---:|---:|---:|---:|---|---:|---|
| button | 0 | 0 | 0 | 0 | N/A (0 valid) | 0 | Unattempted |
| placement | 0 | 0 | 0 | 0 | N/A (0 valid) | 0 | Unattempted |
| pouring | 0 | 0 | 0 | 0 | N/A (0 valid) | 0 | Unattempted |
| tidying | 0 | 0 | 0 | 0 | N/A (0 valid) | 0 | Unattempted |
| shape_matching | 0 | 0 | 0 | 0 | N/A (0 valid) | 0 | Unattempted |
| writing | 0 | 0 | 0 | 0 | N/A (0 valid) | 0 | Unattempted |
| tower | 0 | 0 | 0 | 0 | N/A (0 valid) | 0 | Unattempted |
| measurement | 0 | 0 | 0 | 0 | N/A (0 valid) | 0 | Unattempted |
| bimanual_handover | 0 | 0 | 0 | 0 | N/A (0 valid) | 0 | Unattempted |
| bimanual_tray | 0 | 0 | 0 | 0 | N/A (0 valid) | 0 | Unattempted |
| bimanual_drawer | 0 | 0 | 0 | 0 | N/A (0 valid) | 0 | Unattempted |

Totals: 0 started, 0 completed, 0 valid, 0 successful, 0 infrastructure-invalid
robotics episodes. Success rate is undefined with a zero denominator. There were
two failed inference preflights, counted separately from robotics episodes.
The 55 planned robotics attempts were not executed. No in-chat development pilot,
scripted controller run, or scoring fixture was substituted.

## Inference evidence

The installed client was `codex-cli 0.163.0`. `codex login status` reported
`Logged in using ChatGPT`. Official [OpenAI authentication guidance](https://learn.chatgpt.com/docs/auth)
distinguishes subscription sign-in from API-key usage-based access; the existing
subscription login was used without opening or copying credential files.

1. A sandboxed check failed during local client initialization with an operating
   system permission error, before inference.
2. Outside that sandbox, a default-model request to return exactly `READY` emitted
   thread/turn start events, then inference timeout/retry events. It was stopped
   at 50 seconds without a model response, completed turn, or usage.
3. An attempted HTTP transport override was rejected during configuration because
   this CLI reserves the built-in `openai` provider ID. The override was removed.
4. A corrected check explicitly requested `gpt-6.1-sol`, low reasoning effort,
   and the existing ChatGPT login. It emitted thread/turn start events and three
   error events, with inference timeout/retry categories. It was stopped after
   90.020 seconds without `READY`, turn completion, or usage.

The requested model and reasoning setting are verifiable command settings; the
served model identity is unverified. No model usage was returned, so consumption
is unknown rather than zero. Timeout causes were not established; no account,
model-entitlement, or service-wide outage diagnosis is claimed.

The CLI returned process code zero after termination in both timed-out checks.
A zero process code therefore does not establish inference success. The reusable
preflight requires `READY`, a completed turn, no deadline expiry, no observed tool
execution, and process code zero. Its seven focused transport tests pass, including
this misleading-zero-exit case and rejection of an API-key login.

See [the sanitized preflight sequence](preflight-sequence.json) and
[the explicit-model evidence](inference-preflight.json). The initial raw probe logs
were kept outside Git in a temporary operator directory; the reusable check stores
only allowlisted summaries and never persists raw CLI output.

## Observation boundary and controller support

The allowlisted `artifacts/agent-bundle.zip` was extracted to a new temporary
public working directory. Its SHA-256 was
`f1ad02b62f1d2948b8805db53993aa4272544fce3c9ca04d0f9f300eac1295a5`.
Only the public instructions/client/robot-only IK helper, CLI help, and server
entry-point help were inspected. No evaluator/task implementation, scoring tests,
procedural scenes, private episode material, prior pilots, policies, or trajectories
were read. No credentials or episode tokens were needed for robotics transport.

The proposed controller may use public robot-only IK for model-selected Cartesian
targets. It was not exercised here. Objects must be localized from RGB images;
IK provides robot geometry support, not task state. Dual-arm episodes require
`move_bimanual` and the public `wrist_left`/`wrist_right` cameras. A fresh server
must omit `--seed` and use a new private directory for every episode.

CLI checks ignored user configuration/rules, ran ephemerally in read-only mode,
disabled shell/apps/plugins/memory/multi-agent/browser/computer/image/hook features,
and disabled web search. The corrected reusable check also forces ChatGPT login
and removes API-key/access-token/provider override variables from its child
environment without reading their values. No paid API billing path was used.

No model response arrived, so actual tool removal and isolation were not attested.
Same-host execution and prompt restrictions do not constitute an adversarial
sandbox. Private post-termination scores would require an operator-only
allowlisted projection; none were obtained because no episode began.

There are no robotics screenshots, action/reason logs, or outcome visualizations:
no public episode observation or model-selected physical action occurred.

## Resume prerequisite

First obtain a completed subscription-backed minimal inference response. From a
trusted operator terminal, run:

```sh
python3 scripts/check_subscription_inference.py --model gpt-6.1-sol --timeout 90 --output /private/tmp/robot-inference-preflight.json
```

The script exits 0 only when its transport checks pass and 2 for a reported
preflight block. A pass still requires separate served-model metadata and tool
surface verification before claiming an isolated evaluation. Then execute the
predeclared one-per-family coverage phase using fresh random episodes, log public
observations/model decisions, and expand to 55 only while infrastructure works.
No conclusions about unsolvability or success rate follow from this blocked run.
