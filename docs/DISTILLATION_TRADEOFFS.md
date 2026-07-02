# Distillation Trade-offs: Prompts, Iterations, and Convergence

Notes from distilling a few-step **Wan2.1-1.3B causal student** from a **14B t2v
teacher** (DMD2 + Self-Forcing) for StreamDiffusion-v2 on AWS Trainium. Written
as working notes for a blog on *targeted* video-diffusion distillation.

## What the distillation actually optimizes

DMD (Distribution Matching Distillation) does **not** compare the student's video
to the teacher's video (that's regression, and it produces blur at low step
counts). Instead, per iteration:

1. Student generates a latent `x0` in its few-step schedule (e.g. 2 steps).
2. Noise `x0` to a random timestep → `x_t`.
3. Ask **two** networks for their score of `x_t`:
   - **real score** = the frozen 14B teacher → "the realistic direction"
   - **fake score** = a live 1.3B tracker of the student → "the student's-current direction"
4. Gradient into the student = **(fake − real)**. Push the student from where it
   is toward where realistic is.
5. Separately, the fake_score net trains to keep tracking the student.

**Convergence point:** when the student's few-step output *distribution* matches
the teacher's, `fake ≈ real`, the gradient → 0. Training converges to a fixed
point — **not** "we finished looping the prompts."

**Ceiling:** the student converges *toward the teacher's quality at the student's
step count*. It cannot exceed the teacher. If the teacher makes great
pancake-flip videos, a converged 2-step student makes teacher-quality pancake
videos in 2 steps — no better.

## The two independent knobs

| Knob | Controls | Rule of thumb |
|---|---|---|
| **# prompts** | *breadth* — what the model is good at | you only get what you show it |
| **# iterations** | *convergence* — whether it reached the teacher on that set | ~updates-per-prompt, not total |

These are orthogonal. Prompts set the width of the competence; iterations decide
whether you actually reached the teacher within that width.

## Targeted vs broad: you can choose

| Setup | Result | Cost |
|---|---|---|
| **10 prompts × ~thousands iters** | good on *those 10* (+ nearby motions); useless off-set | cheap |
| **100 prompts × thousands iters** | good on that narrow domain | moderate |
| **20k prompts × thousands iters** | good across the whole kinetics motion distribution | large |
| N prompts × 5 iters | proves plumbing only — barely moved | trivial |

**Key insight for a targeted model:** if you only need N specific prompts to
work, train on exactly those N and cycle them. You do **not** need the full
corpus. This is deliberate overfitting — and for a fixed prompt set that's the
*goal*, not a bug.

## Sizing iterations (the empirical part)

The transferable unit is **gradient-updates-per-prompt**, not total iters. And in
DMD the *generator* (student) only updates every `dfake_gen_update_ratio` steps
(the rest train fake_score), so:

```
student_updates = total_iters / dfake_gen_update_ratio
updates_per_prompt = student_updates / num_prompts
```

- A 1.3B student memorizing a single target wants ~100–300 updates on it.
- For a tiny set you can set `dfake_gen_update_ratio = 1` (student every iter) —
  the careful teacher/generator balance that ratio>1 buys is for large-scale
  runs, not memorization.
- So for **10 prompts**: ~80–150 updates/prompt → order **hundreds-to-low-thousands
  of iters**. The exact number is *discovered*, not computed.

**How to discover it (small-experiment discipline):**
1. Set the prompt set + `save_every` (checkpoint often, e.g. every 200 iters).
2. Watch **two signals together**:
   - `loss_G` trend — should decrease and *flatten* (it's noisy per-iter due to the
     random timestep; watch the trend over ~50 iters).
   - the **checkpoint output video** on an *in-set* prompt — watch where it stops
     improving.
3. Stop at the plateau (loss flat **and** video stops improving).

**Why both signals, never loss alone:** loss going down while video stays bad =
converging to something *wrong* (degenerate teacher signal, gradient too weak).
Loss says "is it moving"; video says "is it moving toward *right*." Neither alone
is sufficient — this bit us repeatedly (a fast/low number on garbage output).

## Can a small run short-circuit the big one?

**Partially — it de-risks and sizes, but does not replace measurement.**

What the 10-prompt run transfers to a 20k run:
- **Proof of convergence** — that DMD converges at all and the gradient is
  sufficient (the biggest unknown, answered cheaply).
- **A per-prompt update budget** — a starting estimate: `total ≈ per_prompt × N`.
- **Known-good knobs** — lr, dfake ratio, teacher signal validated.

Why it is **not** a clean multiply:
1. **Generalization vs memorization.** 10 prompts = pure memorization (many
   updates each). 20k prompts *share structure* (all kinetics motion) → the model
   learns transferable features and needs **fewer** updates/prompt. So
   `per_prompt × 20k` is an **over-estimate**.
2. **Overfitting inverts the goal.** On 10 you *want* memorization; on 20k that
   same per-prompt budget would overfit and hurt generalization → you stop
   *earlier* per prompt.
3. **The loss-curve shape differs.** 10 prompts converge sharply; 20k converge
   gradually — "loss flattened" occurs at a different relative point.

**Verdict:** the small run turns the big run from a blind expensive gamble into a
*tuned* run with a checkpoint-watching stop criterion. You still validate the 20k
by watching checkpoints — you can't skip that, because generalization changes the
per-prompt cost in a direction only measurement reveals.

## Practical recipe (this project)

- Teacher: 14B **t2v** (matches the t2v/v2v student; an i2v teacher would
  misalign scores). Frozen.
- Student: base Wan2.1-1.3B, distilled to the ship step count (1–2 steps = the fps
  lever), through the *real* SDV2 streaming loop (same schedule + rolling KV cache
  + sink tokens + block size as inference) so train == inference. This is what a
  generic causal checkpoint (e.g. RollingForcing) lacked — it was causal but
  mismatched to our pipeline, hence noise.
- Train on Trainium (eager torch_neuronx, native autograd). 3 TP-4 groups
  (teacher / student / fake) to fit HBM; grad only through the last denoise step
  to bound rollout-activation memory.
