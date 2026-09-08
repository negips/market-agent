"""
train_model.jl

Train the StockSwingPredictor MLP on the assembled dataset.
Loads the pre-split, pre-normalised CSVs from build_dataset.jl.
Saves the best model weights and training log.

Usage:
  julia --project=packages/StockSwingPredictor scripts/train_model.jl
  julia --project=packages/StockSwingPredictor scripts/train_model.jl --epochs 200
  julia --project=packages/StockSwingPredictor scripts/train_model.jl --lr 5e-4 --batch 128
"""

using StockSwingPredictor, Flux, JSON3, Dates, Printf

const REPO_ROOT   = joinpath(@__DIR__, "..")
const TRAIN_DIR   = joinpath(REPO_ROOT, "website", "data", "training")
const MODELS_DIR  = joinpath(REPO_ROOT, "website", "data", "models")

function parse_args()
    opts = Dict{String,Any}(
        "epochs"  => 150,
        "lr"      => 1e-3,
        "batch"   => 64,
        "l2"      => 1e-4,
        "patience"=> 15,
    )
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
  --batch N     Batch size (default: 64)
  --l2 FLOAT    L2 regularisation lambda (default: 1e-4)
  --patience N  Early stopping patience (default: 15)

Input:
  website/data/training/dataset_{train,val,test}.csv
  website/data/training/norm_stats.json

Output:
  website/data/models/swing_predictor.bson
  website/data/models/training_log.json
""")
            exit(0)
        elseif a == "--epochs"  opts["epochs"]   = parse(Int,     ARGS[i+1]); i += 2
        elseif a == "--lr"      opts["lr"]        = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--batch"   opts["batch"]     = parse(Int,     ARGS[i+1]); i += 2
        elseif a == "--l2"      opts["l2"]        = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--patience" opts["patience"] = parse(Int,     ARGS[i+1]); i += 2
        else i += 1
        end
    end
    return opts
end

function main()
    opts = parse_args()

    # ── Load datasets ─────────────────────────────────────────────────────────

    train_file = joinpath(TRAIN_DIR, "dataset_train.csv")
    val_file   = joinpath(TRAIN_DIR, "dataset_val.csv")
    test_file  = joinpath(TRAIN_DIR, "dataset_test.csv")
    stats_file = joinpath(TRAIN_DIR, "norm_stats.json")
    vocab_file = joinpath(TRAIN_DIR, "sector_vocab.json")

    for f in (train_file, val_file, test_file, stats_file)
        isfile(f) || error("Not found: $f\nRun: julia scripts/build_dataset.jl")
    end

    @info "Loading datasets…"
    ds_train = load_dataset(train_file)
    ds_val   = load_dataset(val_file)
    ds_test  = load_dataset(test_file)
    norm_stats   = load_norm_stats(stats_file)
    sector_vocab = isfile(vocab_file) ?
                   collect(String, JSON3.read(read(vocab_file, String))) : String[]

    n_features = size(ds_train.X, 1)
    @info "Features: $n_features"
    @info "Train: $(size(ds_train.X, 2))  Val: $(size(ds_val.X, 2))  Test: $(size(ds_test.X, 2))"

    # Label stats (final-hour bar — end-of-day-5 close vs reference)
    final_h = ds_train.y[end, :]
    @printf("Train label (eod day5): mean=%.4f%%  std=%.4f%%  shape=%s\n",
            mean(final_h)*100, std(final_h)*100, string(size(ds_train.y)))

    # ── Build and train model ─────────────────────────────────────────────────

    model = build_model(n_features; dropout_rate=0.3)
    @info "Model parameters: $(sum(length, Flux.params(model)))"

    println()
    @info "Training…"
    model, log = train!(
        model,
        ds_train.X, ds_train.y,
        ds_val.X,   ds_val.y;
        epochs    = opts["epochs"],
        batchsize = opts["batch"],
        lr        = Float32(opts["lr"]),
        l2_lambda = Float32(opts["l2"]),
        patience  = opts["patience"],
    )

    # ── Evaluate on test set ──────────────────────────────────────────────────

    println()
    @info "Test set evaluation:"
    test_metrics = evaluate(model, ds_test.X, ds_test.y)
    merge!(log, test_metrics)
    log["trained_at"] = string(now(UTC))
    log["n_features"]  = n_features
    log["hyperparams"] = opts

    # ── Save ──────────────────────────────────────────────────────────────────

    mkpath(MODELS_DIR)
    model_path = joinpath(MODELS_DIR, "swing_predictor.bson")
    log_path   = joinpath(MODELS_DIR, "training_log.json")

    save_model(model, norm_stats, sector_vocab, model_path;
               meta=Dict("n_features" => n_features,
                         "trained_at" => string(now(UTC)),
                         "test_ic"    => get(test_metrics, "information_coefficient", 0.0)))
    save_training_log(log, log_path)

    println()
    @info "Saved: $model_path"
    @info "Saved: $log_path"
end

main()
