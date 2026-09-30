"""
PPO training loop: rollout → `ppo_update!` → checkpoint, mirroring
`StockSwingPredictor/src/train.jl`'s conventions (checkpoint-on-improvement,
streamed `episode_log.jsonl`, `STOP`/`STOP_NOW` sentinel files) so the two
packages' training scripts feel the same from the REPL/CLI.

One iteration = one full training-episode rollout (stochastic actions, for
exploration) + `ppo_update!` on that episode's buffer. Every `eval_every`
iterations, a held-out episode is rolled out with `greedy=true` (no learning)
purely to track out-of-sample performance — checkpointing follows the
held-out return when a validation episode is configured, and falls back to
the (noisier) training-rollout return otherwise.
"""

using Flux, Dates, Printf, JSON3, Random

const DEFAULT_ITERATIONS      = 200
const DEFAULT_K_EPOCHS        = 4
const DEFAULT_MINIBATCH       = 32
const DEFAULT_LR              = 3f-4
const DEFAULT_EVAL_EVERY      = 10

"""
Train `policy` against `env` (already wrapping the `InferenceCache` to train
on) using `train_config` as the (repeated, per iteration) training episode.

# Arguments
- `val_config`: an `EpisodeConfig` over a held-out date range — its greedy
  rollout return drives checkpointing when given; omit to checkpoint on the
  training rollout return instead (fine for a short sanity run).
- `embed_dim`/`macro_embed_dim`/`attn_heads`/`critic_hidden`: must match
  whatever `policy` was actually constructed with — `ActorCriticPolicy`
  doesn't carry its own hyperparameters as a field, so `train_policy!` needs them
  again here purely to write a reloadable checkpoint (see `save_policy`).
- `stop_file`: `touch <stop_file>` for a clean stop (checkpoint saved) after
  the current iteration; `touch <stop_file with STOP replaced by STOP_NOW>`
  for a hard stop (no save), exactly matching `train_model.jl`.
- `live_path`: when non-empty, streams this run's current episode (portfolio
  value curve, holdings, recent trades) to that JSON path every `live_every_bars`
  bars — see `live.jl` / `website/tradinggamelive.html`. Omit to disable.
- `iteration_offset`: added to every iteration number in logging/printing/
  checkpoint metadata — the loop itself always runs `1:iterations` (i.e.
  `iterations` means "how many more to run"). Lets a resumed run's iteration
  numbers continue from where a previous run left off instead of restarting
  at 1 — same convention as `StockSwingPredictor.train!`'s `epoch_offset`.
  `best_return` itself is NOT seeded from any prior run (also matching
  `StockSwingPredictor`): the first post-resume checkpoint write is
  unconditional, exactly as it is for a fresh run's first improvement.
- `device`: `:cpu` (default) or `:gpu` — moves `policy` there once, up front
  (same convention as `StockSwingPredictor.train!`). `collect_rollout`/
  `ppo_update!` move each batch to match; checkpoints are always written from
  a CPU copy regardless (`save_policy` does this internally), so `policy.bson`
  stays portable across devices either way.

# Returns
`(policy, log)` — `policy` is always reloaded to its best-checkpointed weights
before returning (same convention as `StockSwingPredictor`'s `train!`), even
if the run ended via `STOP`/`STOP_NOW` or the loss/return regressed on the
final iterations. `log` is the same dict written to `episode_log_path`.
"""
function train_policy!(policy::ActorCriticPolicy, env::TradingGameEnv, train_config::EpisodeConfig;
                 val_config::Union{Nothing, EpisodeConfig}=nothing,
                 iterations::Int=DEFAULT_ITERATIONS,
                 k_epochs::Int=DEFAULT_K_EPOCHS,
                 minibatch_size::Int=DEFAULT_MINIBATCH,
                 lr::Float32=DEFAULT_LR,
                 eval_every::Int=DEFAULT_EVAL_EVERY,
                 macro_cache::Union{Nothing, MacroCache}=nothing,
                 news_fn::Function=_zero_news,
                 checkpoint_path::String="",
                 episode_log_path::String="",
                 stop_file::String="",
                 live_path::String="",
                 live_every_bars::Int=5,
                 iteration_offset::Int=0,
                 device::Symbol=:cpu,
                 embed_dim::Int=64, macro_embed_dim::Int=16, attn_heads::Int=4,
                 critic_hidden::Vector{Int}=[64, 32],
                 rng::AbstractRNG=Random.default_rng())
    policy = _to_device(device)(policy)
    opt_state = Flux.setup(Flux.Adam(lr), policy)
    live_tracker = LiveTracker(path=live_path, every_bars=live_every_bars)

    best_return    = -Inf32
    # `cpu(x) === x` for an already-CPU array (no copy at all — verified directly;
    # Flux only copies on a real device transfer) — training here is CPU-only, so
    # without `deepcopy`, `best_state` would alias the live weights and silently
    # "restore" whatever the LAST iteration produced, not the best one.
    best_state     = deepcopy(Flux.state(cpu(policy)))
    stop_now_fired = false

    log = Dict{String, Any}(
        "iterations_run"    => 0,
        "train_return"      => Float32[],
        "train_final_value" => Float32[],
        "loss"               => Float32[],
        "val_return"         => Float32[],
        "val_final_value"    => Float32[],
        "best_return"        => -Inf32,
        "best_iteration"     => 0,
        "started_at"         => string(now(UTC)),
        "checkpoint_path"    => checkpoint_path,
    )

    resume_str = iteration_offset > 0 ? " (resuming from iteration $iteration_offset)" : ""
    @info "Training TradingGame policy: $iterations iterations, $(length(train_config.candidate_universe)) candidates$resume_str"
    train_start = time()

    for iter in 1:iterations
        abs_iter   = iter + iteration_offset
        iter_start = time()

        start_episode!(live_tracker; iteration=abs_iter, phase="train")
        buffer = collect_rollout(env, policy, train_config;
                                  macro_cache=macro_cache, news_fn=news_fn, rng=rng, device=device,
                                  live_cb=make_live_callback(live_tracker))
        train_return = sum(s.reward for s in buffer)
        train_value  = portfolio_value(env)   # env sits at the rollout's terminal state

        start_update!(live_tracker; iteration=abs_iter, k_epochs=k_epochs,
                       total_minibatches=k_epochs * cld(length(buffer), minibatch_size))
        stats = ppo_update!(policy, opt_state, buffer;
                             k_epochs=k_epochs, minibatch_size=minibatch_size, device=device,
                             clip_eps=CLIP_EPS, value_loss_coef=VALUE_LOSS_COEF,
                             entropy_coef=ENTROPY_COEF, gamma=GAMMA, gae_lambda=GAE_LAMBDA, rng=rng,
                             verbose=true, progress_cb=make_update_callback(live_tracker, env))

        push!(log["train_return"], train_return)
        push!(log["train_final_value"], train_value)
        push!(log["loss"], stats["loss"])

        # When val_config is given, checkpointing is driven ONLY by the held-out
        # return — a non-eval iteration has no fresh validation signal, so it
        # must not silently fall back to the (noisier, in-sample) train_return.
        candidate_return = val_config === nothing ? train_return : nothing
        val_return, val_value = nothing, nothing
        do_eval = val_config !== nothing && (iter % eval_every == 0 || iter == iterations)
        if do_eval
            start_episode!(live_tracker; iteration=abs_iter, phase="val")
            eval_buffer = collect_rollout(env, policy, val_config;
                                           macro_cache=macro_cache, news_fn=news_fn,
                                           greedy=true, rng=rng, device=device,
                                           live_cb=make_live_callback(live_tracker))
            val_return = sum(s.reward for s in eval_buffer)
            val_value  = portfolio_value(env)
            push!(log["val_return"], val_return)
            push!(log["val_final_value"], val_value)
            candidate_return = val_return
        end

        improved = candidate_return !== nothing && candidate_return > best_return
        if improved
            best_return           = candidate_return
            best_state            = deepcopy(Flux.state(cpu(policy)))
            log["best_return"]    = best_return
            log["best_iteration"] = abs_iter
            if !isempty(checkpoint_path)
                save_policy(policy, checkpoint_path; embed_dim=embed_dim, macro_embed_dim=macro_embed_dim,
                            attn_heads=attn_heads, critic_hidden=critic_hidden,
                            meta=Dict("checkpoint" => true, "saved_at" => string(now(UTC)),
                                      "iteration" => abs_iter))
            end
        end

        iter_secs = time() - iter_start
        eval_str  = val_return === nothing ? "" : @sprintf(" | val %.4f", val_return)
        @printf("Iter %4d | train_return %.4f | value %.0f | loss %.4f%s | %s%s\n",
                abs_iter, train_return, train_value, stats["loss"], eval_str,
                _fmt_duration(round(Int, iter_secs)), improved ? " ★" : "")

        if !isempty(episode_log_path)
            # `best_return` starts at -Inf32 (nothing has improved on yet) — JSON has
            # no Infinity literal, so JSON3.write errors outright unless it's swapped
            # for `null` first, same convention `_sanitize_json` uses for the full log.
            logged_best = isfinite(best_return) ? best_return : nothing
            open(episode_log_path, "a") do io
                JSON3.write(io, (iteration=abs_iter, train_return=train_return, train_final_value=train_value,
                                 loss=stats["loss"], val_return=val_return, val_final_value=val_value,
                                 best_return=logged_best, improved=improved,
                                 elapsed_secs=round(time() - train_start, digits=1)))
                println(io)
            end
        end

        stop_now_file = isempty(stop_file) ? "" : replace(stop_file, "STOP" => "STOP_NOW")
        if !isempty(stop_now_file) && isfile(stop_now_file)
            rm(stop_now_file)
            @info "STOP_NOW detected — hard interrupt after iteration $abs_iter (no save)"
            stop_now_fired = true
            break
        end
        if !isempty(stop_file) && isfile(stop_file)
            rm(stop_file)
            @info "Stop file detected — clean interrupt after iteration $abs_iter"
            break
        end
    end

    log["iterations_run"] = length(log["train_return"])
    log["stop_now"]       = stop_now_fired
    Flux.loadmodel!(policy, best_state)   # always return the best-checkpointed weights, same convention as StockSwingPredictor
    return policy, log
end

"""Save the training log as JSON. Replaces Inf/NaN with null (JSON-safe),
same convention as `StockSwingPredictor.save_training_log` — named
differently (not `save_training_log`) because both packages export a
function of that name; `using TradingGame, StockSwingPredictor` together
(exactly the combo `scripts/train_trading_policy.jl` uses) makes the bare
name ambiguous otherwise, same reason `train!` is `train_policy!` here."""
function save_policy_training_log(log::Dict, path::String)
    open(path, "w") do io; JSON3.pretty(io, _sanitize_json(log)); end
end

_sanitize_json(x::AbstractFloat)  = isfinite(x) ? x : nothing
_sanitize_json(x::AbstractVector) = [_sanitize_json(v) for v in x]
_sanitize_json(x::Dict)           = Dict(k => _sanitize_json(v) for (k, v) in x)
_sanitize_json(x)                 = x

function _fmt_duration(seconds::Int)::String
    seconds < 60   && return "$(seconds)s"
    seconds < 3600 && return "$(seconds ÷ 60)m $(seconds % 60)s"
    return "$(seconds ÷ 3600)h $(seconds % 3600 ÷ 60)m"
end
