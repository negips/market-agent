"""
train_model.jl

Train the SwingPredictor on the assembled dataset from build_dataset.jl.

Usage:
  julia --project=packages/StockSwingPredictor scripts/train_model.jl
  julia --project=packages/StockSwingPredictor scripts/train_model.jl --epochs 200
  julia --project=packages/StockSwingPredictor scripts/train_model.jl --lr 5e-4 --batch 64
"""

using StockSwingPredictor, Flux, JSON3, Dates, Printf, Statistics

const REPO_ROOT  = joinpath(@__DIR__, "..")
const TRAIN_DIR  = joinpath(REPO_ROOT, "website", "data", "training")
const MODELS_DIR = joinpath(REPO_ROOT, "website", "data", "models")

function parse_args()
    opts = Dict{String,Any}("epochs"=>150, "lr"=>1e-3, "batch"=>32,
                             "l2"=>1e-4, "patience"=>15)
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

Input:  website/data/training/dataset.bson
Output: website/data/models/swing_predictor.bson
        website/data/models/training_log.json
""")
            exit(0)
        elseif a == "--epochs"  ; opts["epochs"]   = parse(Int,     ARGS[i+1]); i += 2
        elseif a == "--lr"      ; opts["lr"]        = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--batch"   ; opts["batch"]     = parse(Int,     ARGS[i+1]); i += 2
        elseif a == "--l2"      ; opts["l2"]        = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--patience"; opts["patience"]  = parse(Int,     ARGS[i+1]); i += 2
        else i += 1
        end
    end
    return opts
end

function main()
    opts = parse_args()

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

    # ── Build and train model ─────────────────────────────────────────────────

    model = build_model(; dropout_rate=0.3)

    n_params = sum(length, Flux.trainables(model))
    @info "Model parameters: $n_params"
    @info "Universe: $(length(dataset.companies)) companies  ×  $(N_MARKET_DAYS) days  ×  $(N_MARKET_CHANNELS) channels"
    println()

    @info "Training…"
    model, log = train!(
        model, dataset, train_idx, val_idx;
        epochs    = opts["epochs"],
        batchsize = opts["batch"],
        lr        = Float32(opts["lr"]),
        l2_lambda = Float32(opts["l2"]),
        patience  = opts["patience"],
    )

    # ── Test evaluation ───────────────────────────────────────────────────────

    println()
    @info "Test set evaluation:"
    test_metrics = evaluate(model, dataset, test_idx)
    merge!(log, test_metrics)
    log["trained_at"]  = string(now(UTC))
    log["n_companies"] = length(dataset.companies)
    log["n_examples"]  = length(dataset.examples)
    log["hyperparams"] = opts

    # ── Save ──────────────────────────────────────────────────────────────────

    mkpath(MODELS_DIR)
    model_path = joinpath(MODELS_DIR, "swing_predictor.bson")
    log_path   = joinpath(MODELS_DIR, "training_log.json")

    save_model(model, dataset.companies, model_path;
               meta=Dict("trained_at"  => string(now(UTC)),
                         "n_companies" => length(dataset.companies),
                         "test_ic"     => get(test_metrics, "information_coefficient", 0.0)))
    save_training_log(log, log_path)

    println()
    @info "Saved → $model_path"
    @info "Saved → $log_path"
end

main()
