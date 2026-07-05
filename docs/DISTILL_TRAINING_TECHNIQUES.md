# Distilling a few-step video student on Trainium — training techniques

Notes for the blog. Audience: ML scientists/engineers. This documents the
memory techniques that let a DMD distillation training loop run on constrained
per-core HBM, and the one detail that is easy to get catastrophically wrong.

## The end-to-end learning loop (the mechanism, plainly)

Every training iteration is the same five steps. The whole thing hinges on step 3.

1. **Forward — predict.** The student runs its input through its weights (W, b)
   and produces an output `x0`.
2. **Loss — measure wrongness.** Compare output to a target → one scalar `loss`.
3. **Backward — compute the gradient.** `loss.backward()` answers, for *every*
   weight: "if I nudge you up a hair, does loss rise or fall, and by how much?"
   That per-weight direction+magnitude is the **gradient**. This is the entire
   mechanism by which the model knows which way to improve. It is computed by
   walking the **autograd graph** — the chain of operations PyTorch records
   *during the forward pass* linking `loss` back to each weight.
4. **Step — move downhill.** `optimizer.step()` does `W ← W − lr · grad`.
5. **Repeat** thousands of times; weights drift to where loss is low.

The **checkpoint** (`model.pt`) is just the saved values of W and b after N steps.
Nothing else — no videos, no history, no per-iteration bookkeeping. The learning
lives entirely in the weights.

### The failure mode that looks like training but isn't

Backward (step 3) walks the graph from `loss` back to each weight. **If any link
in that chain is broken, the walk dead-ends** and every weight *behind* the break
receives **gradient = 0**. Then step 4 computes `W ← W − lr · 0 = W`: the weight
never changes. You can run 800 iterations; every step multiplies by zero. The
model is **frozen** even though the code looks like it is training, the loss
prints fine, and checkpoints save. The only symptom is: the metric never moves,
and the output never changes across checkpoints.

**The one number that detects it: `grad_norm`** — the total magnitude of all
gradients reaching the trainable weights, measured right after `backward()`,
before `step()`:

```python
gnorm = 0.0
for p in model.parameters():
    if p.grad is not None:
        gnorm += float(p.grad.detach().float().pow(2).sum())
gnorm = gnorm ** 0.5
```

- `grad_norm ≈ 0` → the graph is severed; the model is frozen; no learning rate
  or iteration count will ever help. Go fix the graph.
- `grad_norm > 0` but the metric is flat → the gradient flows; the problem is
  step size / loss construction, not connectivity.

Measuring this one scalar separates "broken plumbing" from "bad hyperparameters"
in a single short run. It should be logged from iteration one of any new training
loop, *before* trusting any loss curve.

## The memory technique: run the student twice (no_grad rollout + recompute-with-grad)

### Why it's needed

DMD (Distribution Matching Distillation) needs the student's generated sample
`x0` in three places within one iteration:

1. the **teacher** scores it (the "real" direction),
2. the **critic/fake** scores it (the "student-like" direction),
3. the **student** must backprop the `(fake − real)` signal into its own weights.

The naive implementation keeps the student's full generation graph alive from the
moment it produces `x0`, through both scoring passes, until the final backward.
For a 30-block causal DiT rolling out a video, that retained graph is enormous —
in our case a flat **~22.7 GB of resident activation/graph tensors per core**,
which pinned the core at ~97% occupancy and OOM'd at a fixed iteration regardless
of leak-hunting. (Confirmed from the NRT device-memory dump: the `Tensors`
category was flat at 22.68 GB and *not* growing — a retained graph, not a leak.)

Tensor-parallelism does **not** fix this: TP shards *weights* (matmul width), not
*activations* or the *autograd graph*. Each core still holds the full activation
tensors for its shard. Models are not inherently core-memory-bound — **TP alone
is.** Real large-model training combines TP + FSDP/ZeRO + activation
checkpointing (and sometimes sequence/pipeline parallel).

### The technique: structural graph-free DMD

Split the student's single logical forward into **two physical forwards**:

```python
# (a) ROLLOUT under no_grad — produce x0 for the two scorers. No graph is built,
#     so this costs almost no memory. Detach to be sure nothing is retained.
with torch.no_grad():
    x0_det = student.rollout(prompt, noise=fixed_noise).detach()

# ... teacher scores x0_det, critic scores x0_det, both under no_grad ...
#     (their (fake - real) signal is a plain tensor, no student graph attached)

# (e) RECOMPUTE the student forward WITH grad, immediately before backward, using
#     the SAME fixed noise so the recompute reproduces x0. This rebuilds the graph
#     only for the single backward, then it is freed the instant backward returns.
x0_grad = student.rollout(prompt, noise=fixed_noise)     # grad ON this time
loss_g  = 0.5 * F.mse_loss(x0_grad, (x0_grad - dmd_grad).detach())
loss_g.backward()          # graph lives only here
opt_g.step()
del x0_grad, loss_g        # hard-free the fresh graph
```

The graph now exists only during (e)→backward, not across the whole iteration.
Combined with **per-block NO_REENTRANT activation checkpointing** and **FSDP2
per-block sharding** (params + grads + optimizer state sharded across the student
group), the resident 22.7 GB collapses to a flat, safe working set (~24 GB total
including model+caches, stable every iter).

### The trap this technique introduces (and why grad_norm is mandatory here)

The recompute in (e) is the exact place the autograd graph can silently fail to
reconnect to the trainable weights:

- If the recomputed `x0_grad` is not actually a differentiable function of the
  student's *sharded/checkpoint-wrapped* parameters — e.g. an inference-only code
  path, an in-place cache write on a leaf view, or a module wrapper that returns a
  detached tensor — then `backward()` runs, `loss` looks normal, but
  **grad_norm ≈ 0** and the model is frozen.
- This is *more* likely with the two-forward trick precisely because the forward
  that produced the value the scorers saw (a) is **not** the forward being
  differentiated (e). Any mismatch between them (a stateful cache branch taken in
  one but not the other, a training-vs-eval flag flip under checkpoint recompute,
  a shard that isn't in the autograd path) severs the chain.

So the technique that saves the memory is exactly the technique that can silently
disconnect the gradient. **The safeguard is to log `grad_norm` from iteration
one** — it is the difference between "we cut memory 10× and still train" and "we
cut memory 10× and froze the model without noticing for a week of runs."

> STATUS (open): as of this writing the single-prompt overfit stayed flat across
> 800 iters — a working loop always overfits one example, so this points at the
> DMD update, and a `grad_norm` probe was added to decide between "graph severed"
> (≈0) and "step-size" (>0). Update this section with the measured value and the
> resolution.

## Reusable checklist for a new training loop (before trusting any loss curve)

1. **Log `grad_norm` after `backward()`, before `step()`** — from iteration one.
2. **Overfit a single example first.** A correct loop always drives one example's
   loss down fast. If it can't, the problem is the loop, not the data/capacity/
   iteration count. Do this before any long or multi-example run.
3. **Watch the model's actual output, not just the loss.** A renormalized or
   surrogate loss (common in DMD: the loss is constructed so its *gradient* is the
   DMD direction, which makes the loss *value* roughly constant) can be flat by
   construction and tells you nothing about convergence. Save checkpoints and
   render them; the output artifact is ground truth.
4. **If you split one forward into two (no_grad + recompute), verify the two take
   the identical code path.** Stateful caches, `self.training` flips under
   checkpoint recompute, and wrapper-returned detached tensors are the usual
   severers.
