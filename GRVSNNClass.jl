# ==============================================================================
# Gated Residual Variable Selection Neural Network for Classification
# ==============================================================================
using ARFFFiles, DataFrames
using CSV
using Statistics
using Flux
using Optimisers
using LinearAlgebra
using Zygote
using Flux: params, glorot_uniform, sigmoid, softmax, Dropout, Dense, LayerNorm, elu, hardsigmoid
using Random
using CUDA
using ProgressMeter
using IterTools
using Hyperopt
using Printf
using MLJBase
using MLUtils # For DataLoader
using CategoricalArrays
using BSON
using DelimitedFiles

########## These parameters need to be chosen ###########

# --- Global parameters, declared once as `const` ---
const NUM_REPEATS = 10 # Number of CV-folds
const VAL_SPLIT = 0.2 # Proportion validation data
const epochs = 150 # Maximum number of epochs
const patience = 15 # Patience for early stopping
const nbo = 50 # Number of iterations in the Bayesian Optimization
const SEED_BASE = 123 # Random seed 
const MODEL_SAVE_DIR = "saved_models_and_stats_class" # Directory for saving best models

# --- Define Hyperparameter Search Space ---
const HYPEROPT_SEARCH_SPACE = (
    encoding_size = [2,4,8,16], # Note that large encoding size dramatically expands model size
    dropout_rate = LinRange(0.1, 0.5, 25),
    learning_rate = exp10.(LinRange(-4.0, -2.0, 25)),
    batch_size = [16,32,64,128],
    weight_decay = exp10.(LinRange(-5.0, -1.0, 25))
)

# --- Name of data file for training ---
df_raw = CSV.read("my_data_class.csv", DataFrame) # Target class variable should be in first column

# The test data file is provided in the === Ensemble Prediction on New Test Data === section

# --- DEVICE SELECTION CPU OR GPU ---
const USE_GPU = false

##########################################################


if USE_GPU
    @eval using CUDA
    if !CUDA.functional()
        @warn "CUDA is not functional, falling back to CPU."
        global const USE_GPU = false
    else
        CUDA.allowscalar(false)
    end
end

const device_transfer = if USE_GPU && @isdefined(CUDA) && CUDA.functional()
    gpu
else
    cpu
end

# --- DATA PREPARATION ---
const TARGET_FEATURE_NAME = names(df_raw)[1]
df = hcat(df_raw[!, TARGET_FEATURE_NAME], df_raw[!, 2:end])
rename!(df, 1 => TARGET_FEATURE_NAME)
@info "Assumed target column is the first column: '$TARGET_FEATURE_NAME'."

target_col = df[!, TARGET_FEATURE_NAME]

if eltype(target_col) <: CategoricalValue
    df[!, TARGET_FEATURE_NAME] = parse.(Int, string.(df[!, TARGET_FEATURE_NAME]))
else
    df[!, TARGET_FEATURE_NAME] = parse.(Int, string.(df[!, TARGET_FEATURE_NAME]))
end

for col_name in names(df, Not(TARGET_FEATURE_NAME))
    if eltype(df[!, col_name]) <: Number
        if any(ismissing, df[!, col_name])
            mean_val = mean(skipmissing(df[!, col_name]))
            df[!, col_name] = replace(df[!, col_name], missing => mean_val)
        end
    end
end

const CATEGORICAL_UNIQUE_THRESHOLD = 10
const NUMERIC_FEATURE_NAMES = String[]
const CATEGORICAL_FEATURES_WITH_VOCABULARY = Dict{String, Vector{Any}}()

for col_name in names(df)
    if col_name == TARGET_FEATURE_NAME
        continue
    end
    col_data = df[!, col_name]
    col_type = eltype(col_data)
    non_missing_data = skipmissing(col_data)
    if col_type <: Number
        num_unique = length(unique(non_missing_data))
        if num_unique <= CATEGORICAL_UNIQUE_THRESHOLD && num_unique > 1
            unique_vals = sort(collect(unique(non_missing_data)))
            CATEGORICAL_FEATURES_WITH_VOCABULARY[col_name] = unique_vals
        else
            push!(NUMERIC_FEATURE_NAMES, col_name)
        end
    else
        unique_vals = sort(collect(unique(non_missing_data)))
        CATEGORICAL_FEATURES_WITH_VOCABULARY[col_name] = unique_vals
    end
end

const FEATURE_NAMES = vcat(NUMERIC_FEATURE_NAMES, sort(collect(keys(CATEGORICAL_FEATURES_WITH_VOCABULARY))))
@info "Identified Numeric Features: $(NUMERIC_FEATURE_NAMES)"
@info "Identified Categorical Features: $(collect(keys(CATEGORICAL_FEATURES_WITH_VOCABULARY)))"

numeric_feature_stats_global = Dict{String, NamedTuple{(:mean, :std), Tuple{Float64, Float64}}}()
for fname in NUMERIC_FEATURE_NAMES
    col_data = df[!, fname]
    mean_val = mean(skipmissing(col_data))
    std_val = std(skipmissing(col_data))
    numeric_feature_stats_global[fname] = (mean=mean_val, std=std_val + eps(Float64))
end

function preprocess_data_to_matrix(df::DataFrame)
    num_samples = size(df, 1)
    total_feature_dim = length(NUMERIC_FEATURE_NAMES)
    for vocab in values(CATEGORICAL_FEATURES_WITH_VOCABULARY)
        total_feature_dim += length(vocab)
    end
    X = zeros(Float32, total_feature_dim, num_samples)
    y = Int.(df[!, TARGET_FEATURE_NAME])
    for i in 1:num_samples
        row = df[i, :]
        current_dim = 1
        for fname in NUMERIC_FEATURE_NAMES
            f_val = row[fname]
            stats = get(numeric_feature_stats_global, fname, (mean=0.0, std=1.0))
            val = Float32((ismissing(f_val) ? stats.mean : f_val) - stats.mean) / stats.std
            X[current_dim, i] = val
            current_dim += 1
        end
        for fname in sort(collect(keys(CATEGORICAL_FEATURES_WITH_VOCABULARY)))
            f_val = row[fname]
            vocab = CATEGORICAL_FEATURES_WITH_VOCABULARY[fname]
            vocab_len = length(vocab)
            if !ismissing(f_val)
                idx = findfirst(==(f_val), vocab)
                if idx !== nothing
                    X[current_dim + idx - 1, i] = 1.0f0
                end
            end
            current_dim += vocab_len
        end
    end
    return X, y
end

# --- Preprocessing function for new test data ---
function preprocess_features_to_matrix(df::DataFrame, numeric_stats::Dict, categorical_features::Dict)
    num_samples = size(df, 1)
    total_feature_dim = length(keys(numeric_stats))
    for vocab in values(categorical_features)
        total_feature_dim += length(vocab)
    end
    X = zeros(Float32, total_feature_dim, num_samples)
    for i in 1:num_samples
        row = df[i, :]
        current_dim = 1
        for fname in sort(collect(keys(numeric_stats)))
            f_val = get(row, Symbol(fname), missing)
            stats = numeric_stats[fname]
            val = Float32((ismissing(f_val) ? stats.mean : f_val) - stats.mean) / stats.std
            X[current_dim, i] = val
            current_dim += 1
        end
        for fname in sort(collect(keys(categorical_features)))
            f_val = get(row, Symbol(fname), missing)
            vocab = categorical_features[fname]
            vocab_len = length(vocab)
            if !ismissing(f_val)
                idx = findfirst(==(f_val), vocab)
                if idx !== nothing
                    X[current_dim + idx - 1, i] = 1.0f0
                end
            end
            current_dim += vocab_len
        end
    end
    return X
end

all_inputs, all_labels = preprocess_data_to_matrix(df)

const UNIQUE_LABELS = sort(unique(all_labels))
const NUM_CLASSES = length(UNIQUE_LABELS)
const LABEL_MAP = Dict(val => i - 1 for (i, val) in enumerate(UNIQUE_LABELS))
const REVERSE_LABEL_MAP = Dict(i - 1 => val for (i, val) in enumerate(UNIQUE_LABELS))
all_labels = [LABEL_MAP[label] for label in all_labels]

@info "Detected $NUM_CLASSES unique classes in the target variable, mapped from $UNIQUE_LABELS to 0..$(NUM_CLASSES-1)."

println("Data loaded and preprocessed.")
println("Total samples: $(size(all_inputs, 2))")
println("Total feature dimension: $(size(all_inputs, 1))")


# --- Custom GRVSNN Layers And Helper Functions ---
struct GatedLinearUnit
    linear_gate::Dense
    linear_output::Dense
end
function GatedLinearUnit(input_dim::Int, output_dim::Int)
    return GatedLinearUnit(
        Dense(input_dim, output_dim, identity; init=glorot_uniform),
        Dense(input_dim, output_dim, hardsigmoid; init=glorot_uniform)
    )
end
Flux.@layer GatedLinearUnit
function (glu::GatedLinearUnit)(x::AbstractArray)
    hardsigmoid_output = glu.linear_output(x)
    return glu.linear_gate(x) .* hardsigmoid_output, hardsigmoid_output
end


struct GatedResidualNetwork
    w_glu::GatedLinearUnit
    w_out::Dense
    residual_projection::Union{Dense, typeof(identity)}
    dropout_layer::Dropout
    activation::Function
end
function GatedResidualNetwork(input_dim::Int, output_dim::Int, dropout_rate::Float64, activation::Function=elu)
    residual_proj = (input_dim == output_dim) ? identity : Dense(input_dim, output_dim, identity; init=glorot_uniform)
    return GatedResidualNetwork(
        GatedLinearUnit(input_dim, output_dim),
        Dense(output_dim, output_dim, identity; init=glorot_uniform),
        residual_proj,
        Dropout(dropout_rate),
        activation
    )
end
Flux.@layer GatedResidualNetwork
function (grn::GatedResidualNetwork)(x::AbstractArray)
    glu_output, hardsigmoid_output = grn.w_glu(x)
    activated_output = grn.activation.(glu_output)
    activated_output = grn.dropout_layer(activated_output)
    linear_out = grn.w_out(activated_output)
    linear_out = grn.dropout_layer(linear_out)
    projected_x = grn.residual_projection(x)
    output = linear_out + projected_x
    return output, hardsigmoid_output
end

struct VariableSelection
    feature_projection_layers::NamedTuple
    feature_grns::NamedTuple
    combined_grn::GatedResidualNetwork
    attention_weights_layer::Dense
    output_dim::Int
    dropout_rate::Float64
    activation::Function
    feature_names_order::Vector{Symbol}
end

function VariableSelection(
    input_dims::Dict{String, Int},
    output_dim::Int,
    dropout_rate::Float64,
    activation::Function,
    feature_names::Vector{String}
)
    feature_names_symbols = Symbol.(feature_names)
    proj_pairs = Pair{Symbol, Dense}[]
    grn_pairs = Pair{Symbol, GatedResidualNetwork}[]
    for fname in feature_names
        sym_fname = Symbol(fname)
        current_input_dim = input_dims[fname]
        push!(proj_pairs, sym_fname => Dense(current_input_dim, output_dim, identity; init=glorot_uniform))
        push!(grn_pairs, sym_fname => GatedResidualNetwork(output_dim, output_dim, dropout_rate, activation))
    end
    feature_projection_layers = NamedTuple(proj_pairs)
    feature_grns = NamedTuple(grn_pairs)
    combined_grn = GatedResidualNetwork(output_dim * length(feature_names), output_dim, dropout_rate, activation)
    attention_weights = Dense(output_dim, length(feature_names), identity; init=glorot_uniform)
    return VariableSelection(
        feature_projection_layers,
        feature_grns,
        combined_grn,
        attention_weights,
        output_dim,
        dropout_rate,
        activation,
        feature_names_symbols
    )
end
Flux.@layer VariableSelection

function (vsn::VariableSelection)(x::Dict{String, T}) where T <: AbstractArray
    processed_features_list_and_hardsigmoids = [
        let 
            fname_sym = fname_sym_local
            fname_str = Zygote.@ignore String(fname_sym)
            feature_input = x[fname_str]
            projected_feature = getproperty(vsn.feature_projection_layers, fname_sym)(feature_input)
            getproperty(vsn.feature_grns, fname_sym)(projected_feature)
        end
        for fname_sym_local in vsn.feature_names_order
    ]
    
    processed_features_list = [item[1] for item in processed_features_list_and_hardsigmoids]
    hardsigmoid_outputs_list = [item[2] for item in processed_features_list_and_hardsigmoids]

    if isempty(vsn.feature_names_order)
        some_input_batch_size = size(first(values(x)), 2)
        return zeros(Float32, vsn.output_dim, some_input_batch_size), zeros(Float32, 0, some_input_batch_size), []
    end
    
    combined_input_for_grn = reduce(vcat, processed_features_list)
    combined_grn_output, _ = vsn.combined_grn(combined_input_for_grn)
    
    attention_logits = vsn.attention_weights_layer(combined_grn_output)
    attention_logits_reshaped = reshape(attention_logits, :, size(combined_grn_output, 2))
    attention_weights = softmax(attention_logits_reshaped; dims=1)
    
    output_list = [attention_weights[idx:idx, :] .* processed_features_list[idx] for idx in 1:length(vsn.feature_names_order)]
    output = sum(output_list)
    
    return output, attention_logits_reshaped, hardsigmoid_outputs_list
end


struct ClassifierHead
    dense_layer::Dense
    num_classes::Int
end
function ClassifierHead(input_dim::Int, num_classes::Int)
    if num_classes == 2
        return ClassifierHead(Dense(input_dim, 1, sigmoid; init=glorot_uniform), num_classes)
    else
        return ClassifierHead(Dense(input_dim, num_classes, identity; init=glorot_uniform), num_classes)
    end
end
Flux.@layer ClassifierHead

function (head::ClassifierHead)(x::AbstractArray)
    logits = head.dense_layer(x)
    if head.num_classes > 2
        return softmax(logits; dims=1)
    else
        return logits
    end
end

struct GRNVSNModel
    vsn_layer::VariableSelection
    classifier_head::ClassifierHead
end
function GRNVSNModel(
    input_dims::Dict{String, Int},
    vsn_output_dim::Int,
    dropout_rate::Float64,
    activation_fn::Function,
    feature_names::Vector{String},
    num_classes::Int
)
    vsn = VariableSelection(
        input_dims,
        vsn_output_dim,
        dropout_rate,
        activation_fn,
        feature_names
    )
    classifier = ClassifierHead(vsn_output_dim, num_classes)
    return GRNVSNModel(vsn, classifier)
end
Flux.@layer GRNVSNModel

function (m::GRNVSNModel)(inputs::Dict{String, T}) where T <: AbstractArray
    features, attention_logits, hardsigmoid_outputs = m.vsn_layer(inputs)
    outputs = m.classifier_head(features)
    return outputs, attention_logits, hardsigmoid_outputs
end

function loss(y_pred, y_true, num_classes::Int)
    if num_classes == 2
        return Flux.Losses.binarycrossentropy(reshape(y_pred, :), Float32.(y_true))
    else
        y_true_onehot = Flux.onehotbatch(y_true, 0:num_classes-1) |> cpu
        y_pred_cpu = y_pred |> cpu
        return Flux.Losses.crossentropy(y_pred_cpu, y_true_onehot)
    end
end

function get_batch(inputs_list, labels_list, batch_size)
    num_samples = length(inputs_list)
    indices = randperm(num_samples)
    batches_inputs = []
    batches_labels = []
    device_transfer_local = USE_GPU && CUDA.functional() ? gpu : cpu
    for i in 1:batch_size:num_samples
        batch_indices = indices[i:min(i + batch_size - 1, num_samples)]
        batch_input_dicts = inputs_list[batch_indices]
        batch_labels = labels_list[batch_indices]
        batched_inputs = Dict{String, AbstractArray{Float32}}()
        for fname in FEATURE_NAMES
            feature_data = [d[fname] for d in batch_input_dicts]
            if ndims(feature_data[1]) == 0
                batched_inputs[fname] = Float32.(hcat(feature_data...))
            elseif ndims(feature_data[1]) == 1
                batched_inputs[fname] = Float32.(hcat(feature_data...))
            else
                batched_inputs[fname] = Float32.(cat(feature_data..., dims=ndims(feature_data[1])+1))
            end
        end
        push!(batches_inputs, batched_inputs |> device_transfer_local)
        push!(batches_labels, Int.(batch_labels) |> device_transfer_local)
    end
    return collect(zip(batches_inputs, batches_labels))
end

function train_model_with_hyperparams(
    vsn_output_dim::Int,
    dropout_rate::Float64,
    learning_rate::Float64,
    batch_size::Int,
    weight_decay::Float64,
    train_inputs,
    train_labels,
    val_inputs,
    val_labels
)
    device_transfer_local = USE_GPU && CUDA.functional() ? gpu : cpu
    example_input_dims = Dict{String, Int}()
    for fname in FEATURE_NAMES
        if fname in NUMERIC_FEATURE_NAMES
            example_input_dims[fname] = 1
        elseif haskey(CATEGORICAL_FEATURES_WITH_VOCABULARY, fname)
            example_input_dims[fname] = length(CATEGORICAL_FEATURES_WITH_VOCABULARY[fname])
        end
    end
    model = GRNVSNModel(
        example_input_dims,
        vsn_output_dim,
        dropout_rate,
        elu,
        FEATURE_NAMES,
        NUM_CLASSES
    ) |> device_transfer_local

    optimizer = Optimisers.AdamW(learning_rate, (0.9, 0.999), weight_decay)
    opt_state = Optimisers.setup(optimizer, model)
    train_batches = get_batch(train_inputs, train_labels, batch_size)
    val_batches = get_batch(val_inputs, val_labels, batch_size)
    min_delta = 0.0001
    best_val_loss = Inf
    epochs_no_improve = 0
    for epoch in 1:epochs
        Flux.trainmode!(model)
        for (x_batch, y_batch) in train_batches
            loss_value, gradients = Zygote.withgradient(m -> loss(first(m(x_batch)), y_batch, NUM_CLASSES), model)
            Optimisers.update!(opt_state, model, gradients[1])
        end
        total_val_loss = 0.0
        Flux.testmode!(model)
        for (x_val_batch, y_val_batch) in val_batches
            val_pred_probs, _, _ = model(x_val_batch)
            total_val_loss += loss(val_pred_probs, y_val_batch, NUM_CLASSES)
        end
        final_val_loss = isempty(val_batches) ? 0.0 : total_val_loss / length(val_batches)
        if final_val_loss < best_val_loss - min_delta
            best_val_loss = final_val_loss
            epochs_no_improve = 0
        else
            epochs_no_improve += 1
        end
        if epochs_no_improve >= patience
            break
        end
    end
    return best_val_loss
end

scientific_notation(x::Float64) = begin
    if x == 0.0
        return "0.0"
    else
        exp = floor(Int, log10(abs(x)))
        coeff = x / (10.0^exp)
        return "$(round(coeff, digits=1))e$(exp)"
    end
end

function create_dict_from_matrix(X_matrix, y_vector)
    num_samples = size(X_matrix, 2)
    inputs_list = Vector{Dict{String, Any}}(undef, num_samples)
    labels_list = y_vector
    for i in 1:num_samples
        current_dict = Dict{String, Any}()
        current_dim = 1
        for fname in NUMERIC_FEATURE_NAMES
            current_dict[fname] = X_matrix[current_dim, i]
            current_dim += 1
        end
        for fname in sort(collect(keys(CATEGORICAL_FEATURES_WITH_VOCABULARY)))
            vocab_len = length(CATEGORICAL_FEATURES_WITH_VOCABULARY[fname])
            one_hot_vector = X_matrix[current_dim:(current_dim + vocab_len - 1), i]
            current_dict[fname] = one_hot_vector
            current_dim += vocab_len
        end
        inputs_list[i] = current_dict
    end
    return inputs_list, labels_list
end

function calculate_confusion_matrix_multiclass(y_true::Vector{<:Integer}, y_pred::Vector{<:Integer}, num_classes::Int)
    cm = zeros(Int, num_classes, num_classes)
    for i in 1:length(y_true)
        if y_true[i] >= 0 && y_true[i] < num_classes && y_pred[i] >= 0 && y_pred[i] < num_classes
            cm[y_true[i] + 1, y_pred[i] + 1] += 1
        end
    end
    return cm
end

function calculate_f1_score_multiclass(cm::AbstractMatrix{Int})
    num_classes = size(cm, 1)
    f1_scores = zeros(Float64, num_classes)
    for i in 1:num_classes
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        
        precision = (tp + fp == 0) ? 0.0 : tp / (tp + fp)
        recall = (tp + fn == 0) ? 0.0 : tp / (tp + fn)
        
        f1_scores[i] = (precision + recall == 0) ? 0.0 : 2 * (precision * recall) / (precision + recall)
    end
    return mean(f1_scores)
end


function calculate_auc_roc(y_true::Vector{<:Integer}, y_pred_probs::Vector{Float64})
    pos_count = sum(y_true .== 1)
    neg_count = sum(y_true .== 0)
    if pos_count == 0 || neg_count == 0
        @warn "AUC-ROC cannot be calculated if one class has zero samples. Returning NaN."
        return NaN
    end
    sorted_indices = sortperm(y_pred_probs, rev=true)
    y_true_sorted = y_true[sorted_indices]
    y_pred_probs_sorted = y_pred_probs[sorted_indices]
    tpr_list = [0.0]
    fpr_list = [0.0]
    current_tps = 0
    current_fps = 0
    for i in 1:length(y_true_sorted)
        if y_true_sorted[i] == 1
            current_tps += 1
        else
            current_fps += 1
        end
        if i == length(y_true_sorted) || y_pred_probs_sorted[i] != y_pred_probs_sorted[i+1]
            push!(tpr_list, current_tps / pos_count)
            push!(fpr_list, current_fps / neg_count)
        end
    end
    auc_val = 0.0
    for i in 2:length(fpr_list)
        auc_val += 0.5 * (tpr_list[i] + tpr_list[i-1]) * (fpr_list[i] - fpr_list[i-1])
    end
    return auc_val
end

function evaluate_model_metrics(model::GRNVSNModel, inputs_list::Vector{Dict{String, Any}}, labels_list::Vector{Int}, batch_size::Int)
    Flux.testmode!(model)
    model_on_cpu = cpu(model)
    
    all_pred_probs = Float32[]
    all_true_labels = Int[]
    
    batches = get_batch(inputs_list, labels_list, batch_size)
    p = Progress(length(batches), "Collecting predictions for metrics...")
    for (x_batch, y_batch) in batches
        x_batch_cpu = cpu(x_batch)
        y_batch_cpu = cpu(y_batch)
        pred_probs_batch, _, _ = model_on_cpu(x_batch_cpu)
        
        append!(all_pred_probs, reshape(pred_probs_batch, :))
        
        append!(all_true_labels, Int.(y_batch_cpu))
        next!(p)
    end
    finish!(p)
    
    pred_matrix = if model_on_cpu.classifier_head.num_classes > 2
        reshape(all_pred_probs, model_on_cpu.classifier_head.num_classes, :)
    else
        reshape(all_pred_probs, :)
    end
    
    avg_loss = loss(pred_matrix, all_true_labels, model_on_cpu.classifier_head.num_classes)
    
    binary_predictions = if model_on_cpu.classifier_head.num_classes == 2
        (pred_matrix .> 0.5) .|> Int
    else
        [argmax(col) - 1 for col in eachcol(pred_matrix)]
    end
    
    accuracy_score = sum(binary_predictions .== all_true_labels) / length(all_true_labels)
    
    f1 = NaN
    auc_roc = NaN
    brier = NaN
    
    if model_on_cpu.classifier_head.num_classes == 2
        cm = calculate_confusion_matrix_multiclass(all_true_labels, binary_predictions, 2)
        f1 = calculate_f1_score_multiclass(cm)
        auc_roc = calculate_auc_roc(all_true_labels, Float64.(pred_matrix))
        brier = mean((pred_matrix .- Float32.(all_true_labels)) .^ 2)
    else
        cm = calculate_confusion_matrix_multiclass(all_true_labels, binary_predictions, model_on_cpu.classifier_head.num_classes)
        f1 = calculate_f1_score_multiclass(cm)
    end
    
    return (
        loss = avg_loss,
        accuracy = accuracy_score,
        f1_score = f1,
        auc_roc = auc_roc,
        brier_score = brier
    )
end


function calculate_feature_importance(model::GRNVSNModel, dataset_inputs_list::Vector{Dict{String, Any}}, feature_names::Vector{Symbol}, batch_size::Int)
    Flux.testmode!(model)
    model_on_cpu = cpu(model)
    
    total_samples_processed = 0
    all_attention_logits = Float32[]

    dummy_labels = fill(Int(0), length(dataset_inputs_list))
    batches = get_batch(dataset_inputs_list, dummy_labels, batch_size)
    p = Progress(length(batches), "Calculating importance (Attention-based)...")
    
    for (x_batch, _) in batches
        x_batch_cpu = cpu(x_batch)
        _, raw_attention_scores_batch, _ = model_on_cpu(x_batch_cpu)
        
        append!(all_attention_logits, Float32.(reshape(raw_attention_scores_batch, :)))
        
        total_samples_processed += size(x_batch_cpu[first(keys(x_batch_cpu))], 2)
        next!(p)
    end
    finish!(p)
    
    num_features = length(feature_names)
    logits_matrix = reshape(all_attention_logits, num_features, :)

    weights_matrix = softmax(logits_matrix; dims=1)
    avg_weights = sum(weights_matrix, dims=2) / total_samples_processed

    avg_logits = sum(logits_matrix, dims=2) / total_samples_processed

    final_importance = avg_weights .* abs.(avg_logits)
    
    importance_list = [(feature_names[i], final_importance[i,1]) for i in 1:num_features]
    sort!(importance_list, by=x->x[2], rev=true)
    
    return importance_list
end


function calculate_hardsigmoid_importance(model::GRNVSNModel, dataset_inputs_list::Vector{Dict{String, Any}}, feature_names::Vector{Symbol}, batch_size::Int)
    Flux.testmode!(model)
    model_on_cpu = cpu(model)

    all_hardsigmoid_outputs_by_feature = [Float32[] for _ in 1:length(feature_names)]
    
    dummy_labels = fill(Int(0), length(dataset_inputs_list))
    batches = get_batch(dataset_inputs_list, dummy_labels, batch_size)
    p = Progress(length(batches), "Calculating importance (Hardsigmoid Std. Dev.)...")

    for (x_batch, _) in batches
        x_batch_cpu = cpu(x_batch)
        _, _, hardsigmoid_outputs_list = model_on_cpu(x_batch_cpu)
        
        for i in 1:length(hardsigmoid_outputs_list)
            append!(all_hardsigmoid_outputs_by_feature[i], reshape(hardsigmoid_outputs_list[i], :))
        end
        next!(p)
    end
    finish!(p)

    std_hardsigmoid_per_feature = [std(outputs) for outputs in all_hardsigmoid_outputs_by_feature]
    
    importance_list = [(feature_names[i], std_hardsigmoid_per_feature[i]) for i in 1:length(feature_names)]
    sort!(importance_list, by=x->x[2], rev=true)
    
    return importance_list
end


if !isdir(MODEL_SAVE_DIR)
    mkdir(MODEL_SAVE_DIR)
end

converted_categorical_features = Dict{String, Vector{String}}()
for (key, val) in CATEGORICAL_FEATURES_WITH_VOCABULARY
    converted_categorical_features[key] = string.(val)
end

BSON.bson("$(MODEL_SAVE_DIR)/global_stats.bson",
          feature_stats=numeric_feature_stats_global,
          feature_names=FEATURE_NAMES,
          target_name=TARGET_FEATURE_NAME,
          categorical_features=converted_categorical_features,
          unique_labels=UNIQUE_LABELS,
          label_map=LABEL_MAP,
          reverse_label_map=REVERSE_LABEL_MAP)

final_metrics_list = []
feature_importance_list = []
hardsigmoid_importance_list = []

# --- Training Loop With Random CV-Folds ---
for run_num in 1:NUM_REPEATS
    println("\n==================================================================")
    println("--- Starting Run $run_num/$NUM_REPEATS with a new random seed ---")
    println("==================================================================\n")
    
    Random.seed!(SEED_BASE + run_num)

    num_samples = size(all_inputs, 2)
    num_val = floor(Int, num_samples * VAL_SPLIT)
    num_train = num_samples - num_val
    shuffled_indices = shuffle(1:num_samples)
    train_indices = shuffled_indices[1:num_train]
    val_indices = shuffled_indices[num_train+1:end]

    train_inputs, train_labels = create_dict_from_matrix(all_inputs[:, train_indices], all_labels[train_indices])
    val_inputs, val_labels = create_dict_from_matrix(all_inputs[:, val_indices], all_labels[val_indices])

    println("Data split complete for this run. Training samples: $(length(train_labels)), Validation samples: $(length(val_labels))")


    println("\n--- Starting Hyperparameter Optimization with Hyperopt.jl ---")
    
    ho_result = @hyperopt for i = nbo,
        encoding_size = HYPEROPT_SEARCH_SPACE.encoding_size,
        dropout_rate = HYPEROPT_SEARCH_SPACE.dropout_rate,
        learning_rate = HYPEROPT_SEARCH_SPACE.learning_rate,
        batch_size = HYPEROPT_SEARCH_SPACE.batch_size,
        weight_decay = HYPEROPT_SEARCH_SPACE.weight_decay

        println("\n Hyperopt Trial (Hyperparams: VSN_Dim=$(encoding_size), Dropout=$(round(dropout_rate, digits=3)), LR=$(scientific_notation(learning_rate)), Batch=$(batch_size), WD=$(scientific_notation(weight_decay)))")
        
        val_loss = train_model_with_hyperparams(
            encoding_size,
            dropout_rate,
            learning_rate,
            batch_size,
            weight_decay,
            train_inputs,
            train_labels,
            val_inputs,
            val_labels
        )
        println(" Trial Validation Loss: $(round(val_loss, digits=5))")
        val_loss
    end

    println("\n--- Hyperparameter Optimization with Hyperopt.jl Complete! ---")

    best_overall_hyperparams = Dict(
        :vsn_output_dim => ho_result.minimizer[1],
        :dropout_rate => ho_result.minimizer[2],
        :learning_rate => ho_result.minimizer[3],
        :batch_size => ho_result.minimizer[4],
        :weight_decay => ho_result.minimizer[5]
    )

    println("Best Hyperparameters found by Hyperopt.jl for this run:")
    display(best_overall_hyperparams)
    println("\nBest Validation Loss achieved: $(round(ho_result.minimum, digits=5))")

    TUNED_VSN_OUTPUT_DIM = best_overall_hyperparams[:vsn_output_dim]
    TUNED_DROPOUT_RATE = best_overall_hyperparams[:dropout_rate]
    TUNED_LEARNING_RATE = best_overall_hyperparams[:learning_rate]
    TUNED_BATCH_SIZE = best_overall_hyperparams[:batch_size]
    TUNED_WEIGHT_DECAY = best_overall_hyperparams[:weight_decay]

    println("\n--- Training Final Model with Best Hyperparameters for this run ---")
    
    example_input_dims_final = Dict{String, Int}()
    for fname in FEATURE_NAMES
        if fname in NUMERIC_FEATURE_NAMES
            example_input_dims_final[fname] = 1
        elseif haskey(CATEGORICAL_FEATURES_WITH_VOCABULARY, fname)
            example_input_dims_final[fname] = length(CATEGORICAL_FEATURES_WITH_VOCABULARY[fname])
        end
    end

    final_model = GRNVSNModel(
        example_input_dims_final,
        TUNED_VSN_OUTPUT_DIM,
        TUNED_DROPOUT_RATE,
        elu,
        FEATURE_NAMES,
        NUM_CLASSES
    ) |> device_transfer

    final_optimizer = Optimisers.AdamW(TUNED_LEARNING_RATE, (0.9, 0.999), TUNED_WEIGHT_DECAY)
    final_opt_state = Optimisers.setup(final_optimizer, final_model)

    final_train_batches = get_batch(train_inputs, train_labels, TUNED_BATCH_SIZE)
    final_val_batches = get_batch(val_inputs, val_labels, TUNED_BATCH_SIZE)

    patience_final = patience
    epochs_final = epochs
    min_delta_final = 0.00005
    best_final_val_loss = Inf
    epochs_no_improve_final = 0
    best_final_model_params = deepcopy(cpu(final_model))

    p = Progress(epochs_final, "Training final model for run $run_num...")
    for epoch in 1:epochs_final
        Flux.trainmode!(final_model)
        for (x_batch, y_batch) in final_train_batches
            loss_value, gradients = Zygote.withgradient(m -> loss(first(m(x_batch)), y_batch, NUM_CLASSES), final_model)
            Optimisers.update!(final_opt_state, final_model, gradients[1])
        end
        total_val_loss = 0.0
        Flux.testmode!(final_model)
        for (x_val_batch, y_val_batch) in final_val_batches
            val_pred_probs, _, _ = final_model(x_val_batch)
            total_val_loss += loss(val_pred_probs, y_val_batch, NUM_CLASSES)
        end
        avg_val_loss = isempty(final_val_batches) ? 0.0 : total_val_loss / length(final_val_batches)
        if avg_val_loss < best_final_val_loss - min_delta_final
            best_final_val_loss = avg_val_loss
            epochs_no_improve_final = 0
            best_final_model_params = deepcopy(cpu(final_model))
        else
            epochs_no_improve_final += 1
        end
        next!(p)
        if epochs_no_improve_final >= patience_final
            println("Early stopping triggered at epoch $epoch for final model training.")
            break
        end
    end
    finish!(p)
    
    BSON.bson("$(MODEL_SAVE_DIR)/best_model_run_$(run_num).bson", model=best_final_model_params)

    println("\n--- Evaluation on Validation Set for Run $run_num ---")
    val_metrics = evaluate_model_metrics(cpu(best_final_model_params), val_inputs, val_labels, TUNED_BATCH_SIZE)
    push!(final_metrics_list, val_metrics)
    @printf "%-15s | %-12s | %-12s | %-12s\n" "Metric" "Value" " " " "
    @printf "%-15s | %-12.5f\n" "Loss" val_metrics.loss
    @printf "%-15s | %-12.5f\n" "Accuracy" val_metrics.accuracy
    @printf "%-15s | %-12.5f\n" "F1-Score" val_metrics.f1_score
    if !isnan(val_metrics.auc_roc)
        @printf "%-15s | %-12.5f\n" "AUC-ROC" val_metrics.auc_roc
        @printf "%-15s | %-12.5f\n" "Brier" val_metrics.brier_score
    end
    
    println("\n--- Calculate Feature Importance for Run $run_num ---")
    attn_importance = calculate_feature_importance(cpu(best_final_model_params), val_inputs, Symbol.(FEATURE_NAMES), TUNED_BATCH_SIZE)
    hs_importance = calculate_hardsigmoid_importance(cpu(best_final_model_params), val_inputs, Symbol.(FEATURE_NAMES), TUNED_BATCH_SIZE)
    push!(feature_importance_list, attn_importance)
    push!(hardsigmoid_importance_list, hs_importance)

    println("Run $run_num complete.")

end

# --- Aggregate and Print Final Results ---
println("\n\n--- Final Aggregated Model Performance Summary (over $NUM_REPEATS runs) ---")
println("------------------------------------------------------------------")
@printf "%-15s | %-12s | %-12s\n" "Metric" "Mean Score" "Std. Dev."

mean_loss = mean([m.loss for m in final_metrics_list])
std_loss = std([m.loss for m in final_metrics_list])
@printf "%-15s | %-12.5f | %-12.5f\n" "Validation Loss" mean_loss std_loss

mean_acc = mean([m.accuracy for m in final_metrics_list])
std_acc = std([m.accuracy for m in final_metrics_list])
@printf "%-15s | %-12.5f | %-12.5f\n" "Validation Acc." mean_acc std_acc

mean_f1 = mean([m.f1_score for m in final_metrics_list])
std_f1 = std([m.f1_score for m in final_metrics_list])
@printf "%-15s | %-12.5f | %-12.5f\n" "Validation F1" mean_f1 std_f1

if NUM_CLASSES == 2
    mean_auc = mean([m.auc_roc for m in final_metrics_list])
    std_auc = std([m.auc_roc for m in final_metrics_list])
    @printf "%-15s | %-12.5f | %-12.5f\n" "Validation AUC" mean_auc std_auc

    mean_brier = mean([m.brier_score for m in final_metrics_list])
    std_brier = std([m.brier_score for m in final_metrics_list])
    @printf "%-15s | %-12.5f | %-12.5f\n" "Validation Brier" mean_brier std_brier
end


# --- Aggregate and Print Feature Importance Results ---
println("\n\n--- Final Aggregated Feature Importance Summary (over $NUM_REPEATS runs) ---")
println("------------------------------------------------------------------")

function aggregate_importance(importance_list)
    summed_scores = Dict{Symbol, Float64}()
    for run_results in importance_list
        for (feature, score) in run_results
            summed_scores[feature] = get(summed_scores, feature, 0.0) + score
        end
    end
    avg_importance = [(feature, score / length(importance_list)) for (feature, score) in summed_scores]
    sort!(avg_importance, by=x->x[2], rev=true)
    return avg_importance
end

avg_attention_importance = aggregate_importance(feature_importance_list)
println("\nAttention-Based Importance (Top $(min(20, length(avg_attention_importance)))):")
@printf "%-20s | %-15s\n" "Feature" "Mean Score"
for (feature, score) in avg_attention_importance[1:min(20, end)]
    @printf "%-20s | %-15.5f\n" String(feature) score
end

avg_hardsigmoid_importance = aggregate_importance(hardsigmoid_importance_list)
println("\nHardsigmoid Std. Dev. Importance (Top $(min(20, length(avg_hardsigmoid_importance)))):")
@printf "%-20s | %-15s\n" "Feature" "Mean Score"
for (feature, score) in avg_hardsigmoid_importance[1:min(20, end)]
    @printf "%-20s | %-15.5f\n" String(feature) score
end

println("\nProcess complete.")

# ------------------------------------------------------------------
# === Ensemble Prediction on New Test Data ===
# ------------------------------------------------------------------
println("\n\n--- Ensemble Prediction on New Test Data (Classification) ---")

try
    # Load the new test data
    # Replace "my_data_class_test.csv" with your actual test filename
    # The data set should NOT have the target variable column
    df_new_test = CSV.read("my_data_class_test.csv", DataFrame)
    
    # Load the saved global stats from the training data
    global_stats_file = BSON.load("$(MODEL_SAVE_DIR)/global_stats.bson")
    target_names = global_stats_file[:unique_labels]
    numeric_feature_stats_for_new = global_stats_file[:feature_stats]
    categorical_features_for_new = global_stats_file[:categorical_features]
    
    # Preprocess the new test data using training set statistics
    println("Preprocessing new data using training set statistics...")
    X_new_test_matrix = preprocess_features_to_matrix(
        df_new_test, 
        numeric_feature_stats_for_new,
        categorical_features_for_new
    )
    num_new_samples = size(X_new_test_matrix, 2)
    
    # Create_dict_from_matrix needs a y_matrix, even if it's empty
    new_data_inputs_list, _ = create_dict_from_matrix(X_new_test_matrix, zeros(Float32, NUM_CLASSES, num_new_samples))
    
    # Initialize an array to store predictions from each model
    all_predictions = zeros(Float32, NUM_CLASSES, num_new_samples, NUM_REPEATS)

    println("Starting ensemble prediction using $NUM_REPEATS models...")
    p = Progress(NUM_REPEATS, "Making predictions with ensemble...")
    for run_num in 1:NUM_REPEATS
        # Load the saved model for the current run
        model_file = BSON.load("$(MODEL_SAVE_DIR)/best_model_run_$(run_num).bson")
        current_model = model_file[:model]
        
        Flux.testmode!(current_model)
        
        test_batches = get_batch(new_data_inputs_list, zeros(Int, num_new_samples), 64)
        run_predictions_list = []
        for (x_batch, _) in test_batches
            pred_batch, _, _ = current_model(x_batch)
            push!(run_predictions_list, cpu(pred_batch))
        end
        all_predictions[:, :, run_num] = hcat(run_predictions_list...)
        
        next!(p)
    end
    finish!(p)

    # Average the *probabilities* from all models (Soft Voting)
    ensemble_predictions_probabilities = dropdims(mean(all_predictions, dims=3), dims=3)
    
    # Convert the averaged probabilities to class labels
    predicted_classes = Flux.onecold(ensemble_predictions_probabilities, target_names)

    # Save the final predictions to a CSV file
    output_filename = "ensemble_predictions_classification.csv"
    open(output_filename, "w") do io
        writedlm(io, ["Predicted Class"], ',')
        writedlm(io, predicted_classes, ',')
    end
    println("\nEnsemble predictions saved to file: $output_filename")
    
    println("\n--- Final Ensemble Predictions ---")
    df_predictions = DataFrame(Predicted_Class = predicted_classes)
    display(df_predictions)
    
catch e
    println("\nAn error occurred during ensemble prediction:")
    println(e)
end