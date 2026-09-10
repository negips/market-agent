"""
train_model.jl

Train the SwingPredictor on the assembled dataset from build_dataset.jl.

Usage:
  julia --project=packages/StockSwingPredictor scripts/train_model.jl
  julia --project=packages/StockSwingPredictor scripts/train_model.jl --epochs 200
  julia --project=packages/StockSwingPredictor scripts/train_model.jl --lr 5e-4 --batch 64
  julia --project=packages/StockSwingPredictor scripts/train_model.jl --resume
  julia --project=packages/StockSwingPredictor scripts/train_model.jl --resume --epochs 50 --lr 1e-4
"""

using StockSwingPredictor, Flux, JSON3, Dates, Printf, Statistics

const REPO_ROOT  = joinpath(@__DIR__, "..")
const CACHE_FILE = joinpath(REPO_ROOT, "website", "data", "inference_cache.bson")
const TRAIN_DIR  = joinpath(REPO_ROOT, "website", "data", "training")
const MODELS_DIR = joinpath(REPO_ROOT, "website", "data", "models")

function parse_args()
    opts = Dict{String,Any}("epochs"=>150, "lr"=>1e-3, "batch"=>32,
                             "l2"=>1e-4, "patience"=>15, "resume"=>false)
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a in ("--help", "-h")
            println("""
Usage:
  julia --project=packages/StockSwingPredictor scripts/train_model.jl [options]

Options:
  --epochs N    Training epochs (default: 150)
  --lr FLOAT    Learning rate (default: 1e-3)
  --batch N     Batch size (default: 32)
  --l2 FLOAT    L2 regularisation lambda (default: 1e-4)
  --patience N  Early stopping patience (default: 15)
  --resume      Load existing weights and continue training from that checkpoint.
                Useful for additional epochs or fine-tuning on new data.

Input:  website/data/training/dataset.bson
        website/data/inference_cache.bson
Output: website/data/models/{arch.name}/swing_predictor.bson
        website/data/models/{arch.name}/training_log.json
        website/data/models/{arch.name}/model_card.json
""")
            exit(0)
        elseif a == "--epochs"  ; opts["epochs"]   = parse(Int,     ARGS[i+1]); i += 2
        elseif a == "--lr"      ; opts["lr"]        = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--batch"   ; opts["batch"]     = parse(Int,     ARGS[i+1]); i += 2
        elseif a == "--l2"      ; opts["l2"]        = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--patience"; opts["patience"]  = parse(Int,     ARGS[i+1]); i += 2
        elseif a == "--resume"  ; opts["resume"]    = true;                       i += 1
        else i += 1
        end
    end
    return opts
end

function main()
    opts = parse_args()

    @info "Loading inference cache…"
    cache = load_inference_cache(CACHE_FILE)
    @info "  $(length(cache.dates)) dates × $(length(cache.companies)) companies"

    dataset_file = joinpath(TRAIN_DIR, "dataset.bson")
    isfile(dataset_file) || error("Not found: $dataset_file\nRun: julia scripts/build_dataset.jl")

    @info "Loading dataset…"
    dataset = load_dataset(dataset_file)
    @info dataset

    train_idx, val_idx, test_idx = time_split(dataset)
    @info "Split: $(length(train_idx)) train / $(length(val_idx)) val / $(length(test_idx)) test"

    final_h = [ex.label[end] for ex in dataset.examples[train_idx]]
    @printf("Train label (eod day5): mean=%.4f%%  std=%.4f%%\n",
            mean(final_h)*100, std(final_h)*100)

    # ── Build or resume model ─────────────────────────────────────────────────

    arch = DUAL_CNN_V1   # swap this line to try a different architecture

    model_dir        = joinpath(MODELS_DIR, arch.name)
    existing_model   = joinpath(model_dir, "swing_predictor.bson")
    existing_card    = joinpath(model_dir, "model_card.json")

    mkpath(model_dir)   # must exist before first checkpoint write during training

    if opts["resume"]
        isfile(existing_model) ||
            error("--resume requested but no checkpoint found at $existing_model")
        model, _, _ = load_model(existing_model)
        prior_epochs = isfile(existing_card) ?
            get(JSON3.read(read(existing_card, String)), :training, Dict())["epochs_run"] : "?"
        @info "Resumed $(arch.name) from checkpoint (prior epochs_run: $prior_epochs)"
    else
        model = build_model(arch)
        @info "Fresh model: $(arch.name)"
    end

    n_params = sum(length, Flux.trainables(model))
    @info "Parameters: $n_params"
    @info "Market context: $(N_MARKET_COMPANIES) companies × $(N_MARKET_DAYS) days"
    println()

    # ── Train ─────────────────────────────────────────────────────────────────

    @info "Training…"
    @info "Checkpoint → $existing_model  (saved on every val improvement)"
    model, log = train!(
        model, dataset, cache, train_idx, val_idx;
        epochs          = opts["epochs"],
        batchsize       = opts["batch"],
        lr              = Float32(opts["lr"]),
        l2_lambda       = Float32(opts["l2"]),
        patience        = opts["patience"],
        checkpoint_path = existing_model,
    )

    # ── Test evaluation ───────────────────────────────────────────────────────

    println()
    @info "Test set evaluation:"
    test_metrics = evaluate(model, dataset, cache, test_idx)
    merge!(log, test_metrics)
    log["trained_at"]  = string(now(UTC))
    log["resumed"]     = opts["resume"]
    log["n_companies"] = length(dataset.companies)
    log["n_examples"]  = length(dataset.examples)
    log["hyperparams"] = opts

    # ── Save ──────────────────────────────────────────────────────────────────

    model_path = existing_model
    log_path   = joinpath(model_dir, "training_log.json")
    card_path  = joinpath(model_dir, "model_card.json")

    save_model(model, dataset.companies, model_path;
               meta=Dict("trained_at"  => string(now(UTC)),
                         "n_companies" => length(dataset.companies),
                         "test_ic"     => get(test_metrics, "information_coefficient", 0.0)))
    save_training_log(log, log_path)
    _save_model_card(card_path, arch, model, dataset, log, test_metrics, opts)

    println()
    @info "Saved → $model_path"
    @info "Saved → $log_path"
    @info "Saved → $card_path"
end

function _save_model_card(path, arch, model, dataset, log, test_metrics, opts)
    n_params = sum(length, Flux.trainables(model))
    ex = dataset.examples
    card = Dict(
        "name"         => arch.name,
        "trained_at"   => log["trained_at"],
        "resumed"      => get(opts, "resume", false),
        "architecture" => Dict(
            "type"                => "DualCNN",
            "market_channels"     => arch.market_channels,
            "market_kernel"       => arch.market_kernel,
            "hourly_channels"     => arch.hourly_channels,
            "hourly_kernel_large" => arch.hourly_kernel_large,
            "hourly_kernel_small" => arch.hourly_kernel_small,
            "mlp_hidden"          => arch.mlp_hidden,
            "dropout_rate"        => arch.dropout_rate,
            "n_params"            => n_params,
        ),
        "inputs" => Dict(
            "market_days"      => N_MARKET_DAYS,
            "market_channels"  => N_MARKET_CHANNELS,
            "market_companies" => N_MARKET_COMPANIES,
            "hourly_bars"      => N_HOURLY_BARS,
            "llm_features"     => N_LLM_FEATURES,
            "pred_hours"       => N_PRED_HOURS,
        ),
        "dataset" => Dict(
            "n_companies" => length(dataset.companies),
            "n_examples"  => length(ex),
            "date_range"  => isempty(ex) ? "" : "$(ex[1].date) → $(ex[end].date)",
        ),
        "training" => Dict(
            "epochs_run"    => log["epochs_run"],
            "best_epoch"    => log["best_epoch"],
            "stopped_early" => log["stopped_early"],
            "best_val_mse"  => log["best_val_mse"],
            "lr"            => opts["lr"],
            "batch"         => opts["batch"],
            "l2"            => opts["l2"],
            "patience"      => opts["patience"],
        ),
        "metrics" => Dict(
            "test_mse"                => get(test_metrics, "test_mse",                nothing),
            "test_mae"                => get(test_metrics, "test_mae",                nothing),
            "direction_accuracy"      => get(test_metrics, "direction_accuracy",      nothing),
            "information_coefficient" => get(test_metrics, "information_coefficient", nothing),
        ),
    )
    open(path, "w") do io; JSON3.pretty(io, card); end
end

main()
