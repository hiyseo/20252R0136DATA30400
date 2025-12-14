#!/bin/bash

################################################################################
# run.sh - Complete Pipeline Execution Script
# Author: 20252R0136
# Description: Executes the entire hierarchical classification pipeline
#
# Usage:
#   ./run.sh                           # Run full pipeline
#   ./run.sh --step 1                  # Run specific step only
#   ./run.sh --skip-labels             # Skip label generation
#   ./run.sh --model-type focal_loss   # Use focal loss model
#
# Steps:
#   0. Environment check
#   1. Data preprocessing
#   2. Silver label generation
#   3. Model training (2-stage: BCE → Self-training)
#   3.5. Model evaluation (using test silver labels)
#   4. Prediction generation (test set)
#   5. Submission file creation (Kaggle format)
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Check if virtual environment is activated
check_venv() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_warning "Virtual environment not activated"
        if [ -d "data304" ]; then
            print_step "Activating virtual environment..."
            source data304/bin/activate
            print_success "Virtual environment activated"
        else
            print_error "Virtual environment 'data304' not found"
            echo "Please run: python3 -m venv data304 && source data304/bin/activate"
            exit 1
        fi
    else
        print_success "Virtual environment active: $VIRTUAL_ENV"
    fi
}

# Check if data exists
check_data() {
    if [ ! -d "data/raw/Amazon_products" ]; then
        print_error "Data directory not found: data/raw/Amazon_products"
        echo "Please ensure dataset is downloaded and placed in correct location"
        exit 1
    fi
    print_success "Data directory found"
}

# Parse command line arguments
SKIP_LABELS=false
SKIP_TRAINING=false
SKIP_EVALUATION=false
SKIP_PREDICTION=false
SKIP_SUBMISSION=false
MODEL_TYPE="baseline"
RUN_STEP=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --step)
            RUN_STEP="$2"
            shift 2
            ;;
        --skip-labels)
            SKIP_LABELS=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        --skip-prediction)
            SKIP_PREDICTION=true
            shift
            ;;
        --skip-submission)
            SKIP_SUBMISSION=true
            shift
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --step NUMBER       Run specific step only (0-5)"
            echo "  --skip-labels       Skip silver label generation (step 2)"
            echo "  --skip-training     Skip model training (step 3)"
            echo "  --skip-evaluation   Skip model evaluation (step 3.5)"
            echo "  --skip-prediction   Skip prediction generation (step 4)"
            echo "  --skip-submission   Skip submission file creation (step 5)"
            echo "  --model-type TYPE   Model type (default: baseline)"
            echo "  --help              Show this help message"
            echo ""
            echo "Steps:"
            echo "  0. Environment check"
            echo "  1. Data preprocessing"
            echo "  2. Silver label generation"
            echo "  3. Model training (2-stage: BCE → Self-training)"
            echo "  3.5. Model evaluation (using test silver labels)"
            echo "  4. Prediction generation (test set)"
            echo "  5. Submission file creation"
            echo ""
            echo "Examples:"
            echo "  ./run.sh                           # Run full pipeline"
            echo "  ./run.sh --step 3                  # Run training only"
            echo "  ./run.sh --skip-labels             # Skip label generation"
# Step 0: Environment check (always run)
if should_run_step "0" || [ -z "$RUN_STEP" ]; then
    print_header "STEP 0: Environment Check"
    check_venv
    check_data
fi

# Step 1: Data preprocessing
if should_run_step "1"; then
    print_header "STEP 1: Data Preprocessing"
    print_step "Loading and preprocessing data..."
    if python3 src/data_preprocessing.py; then
        print_success "Data preprocessing completed"
        print_success "Loaded: 29,487 train + 19,658 test samples"
    else
        print_error "Data preprocessing failed"
        exit 1
    fi
    [ -n "$RUN_STEP" ] && exit 0
fine

# Function to check if step should run
should_run_step() {
    local step=$1
    if [ -n "$RUN_STEP" ]; then
        [ "$RUN_STEP" = "$step" ]
    else
        true
    fi
}

################################################################################
# Main Pipeline
################################################################################

print_header "HIERARCHICAL MULTI-LABEL CLASSIFICATION PIPELINE"
echo "Model Type: $MODEL_TYPE"
echo "Student ID: 20252R0136"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Step 0: Environment check
print_header "STEP 0: Environment Check"
check_venv
check_data

# Step 1: Data preprocessing
print_header "STEP 1: Data Preprocessing"
print_step "Loading and preprocessing data..."
if python3 src/data_preprocessing.py; then
# Step 2: Silver label generation
if should_run_step "2"; then
    if [ "$SKIP_LABELS" = false ]; then
        print_header "STEP 2: Silver Label Generation"
        print_step "Generating silver labels with hybrid top-down approach..."
        print_step "Method: 0.3×keyword + 0.7×semantic + hierarchy filtering"
        print_step "Model: sentence-transformers/all-mpnet-base-v2"
        
        if python3 scripts/generate_labels.py; then
            print_success "Silver labels generated successfully"
            
            # Check generated files
            if [ -f "data/intermediate/train_silver_labels.pkl" ] && [ -f "data/intermediate/test_silver_labels.pkl" ]; then
                print_success "Train labels: data/intermediate/train_silver_labels.pkl"
                print_success "Test labels:  data/intermediate/test_silver_labels.pkl"
            else
                print_error "Label files not found"
                exit 1
            fi
        else
            print_error "Silver label generation failed"
            exit 1
        fi
    else
        print_warning "Skipping silver label generation (--skip-labels)"
        
        # Check if labels exist
        if [ ! -f "data/intermediate/train_silver_labels.pkl" ]; then
            print_error "Silver labels not found. Remove --skip-labels to generate them."
            exit 1
# Step 3: Model training
if should_run_step "3"; then
    if [ "$SKIP_TRAINING" = false ]; then
        print_header "STEP 3: Model Training (2-Stage)"
        print_step "Stage 1: BCE initialization (2 epochs)"
        print_step "Stage 2: Self-training with KLD (3 iterations)"
        print_step "Model type: $MODEL_TYPE"
        print_step "Training data: Train (29,487) + Test (19,658) with silver labels"
        
        if python3 scripts/train_with_config.py; then
            print_success "Training completed successfully"
            
            # Check model file
            MODEL_DIR="data/models/$MODEL_TYPE"
            if [ -f "$MODEL_DIR/best_model.pt" ]; then
                print_success "Model saved: $MODEL_DIR/best_model.pt"
                
                # Show model size
                MODEL_SIZE=$(du -h "$MODEL_DIR/best_model.pt" | cut -f1)
                echo "Model size: $MODEL_SIZE"
                
                # Check training artifacts
                if [ -f "$MODEL_DIR/training_history.json" ]; then
                    print_success "Training history: $MODEL_DIR/training_history.json"
                fi
                
                if ls results/training/$MODEL_TYPE/*.png 1> /dev/null 2>&1; then
                    print_success "Training plots: results/training/$MODEL_TYPE/"
                fi
            else
                print_error "Model file not found"
                exit 1
            fi
        else
            print_error "Training failed"
            exit 1
        fi
    else
        print_warning "Skipping training (--skip-training)"
        
        # Check if model exists
        if [ ! -f "data/models/$MODEL_TYPE/best_model.pt" ]; then
            print_error "Model not found: data/models/$MODEL_TYPE/best_model.pt"
            print_error "Remove --skip-training to train the model."
            exit 1
        fi
    fi
    [ -n "$RUN_STEP" ] && exit 0
fi

# Step 3.5: Model evaluation
if should_run_step "3.5"; then
    if [ "$SKIP_EVALUATION" = false ]; then
        print_header "STEP 3.5: Model Evaluation"
        print_step "Evaluating trained model on test set with silver labels..."
        print_step "Using model: data/models/$MODEL_TYPE/best_model.pt"
        print_step "Note: Metrics are computed using silver labels (pseudo ground truth)"
        
        if python3 src/evaluation/evaluate_model.py \
            --model_path "data/models/$MODEL_TYPE/best_model.pt" \
            --model_type "$MODEL_TYPE" \
            --test_labels_path "data/intermediate/test_silver_labels.pkl" \
            --threshold 0.5; then
            print_success "Evaluation completed successfully"
            
            # Check evaluation outputs
            EVAL_DIR="results/evaluation/$MODEL_TYPE"
            if [ -f "$EVAL_DIR/evaluation_metrics.json" ]; then
                print_success "Metrics saved: $EVAL_DIR/evaluation_metrics.json"
            fi
            
            if ls "$EVAL_DIR"/*.png 1> /dev/null 2>&1; then
                PNG_COUNT=$(ls "$EVAL_DIR"/*.png 2>/dev/null | wc -l)
                print_success "Generated $PNG_COUNT visualization plots in $EVAL_DIR/"
            fi
            
            # Show key metrics
            if command -v jq &> /dev/null && [ -f "$EVAL_DIR/evaluation_metrics.json" ]; then
                echo ""
                echo "Key Metrics:"
                echo "  Micro F1:  $(jq -r '.micro_f1' "$EVAL_DIR/evaluation_metrics.json")"
                echo "  Macro F1:  $(jq -r '.macro_f1' "$EVAL_DIR/evaluation_metrics.json")"
                echo "  Top-3 Acc: $(jq -r '.top_3_accuracy' "$EVAL_DIR/evaluation_metrics.json")"
            fi
        else
            print_error "Evaluation failed"
            exit 1
        fi
    else
        print_warning "Skipping evaluation (--skip-evaluation)"
    fi
    [ -n "$RUN_STEP" ] && exit 0
fi

# Step 4: Prediction generation
if should_run_step "4"; then
    if [ "$SKIP_PREDICTION" = false ]; then
        print_header "STEP 4: Prediction Generation"
        print_step "Generating predictions on test set (19,658 samples)..."
        print_step "Using model: data/models/$MODEL_TYPE/best_model.pt"
        
        if python3 src/inference/predict.py \
            --model_path "data/models/$MODEL_TYPE/best_model.pt" \
            --model_name "$MODEL_TYPE"; then
            print_success "Predictions generated successfully"
            
            # Find latest prediction file
            LATEST_PRED=$(ls -t results/predictions/${MODEL_TYPE}_*.pkl 2>/dev/null | head -1)
            if [ -n "$LATEST_PRED" ]; then
                print_success "Predictions saved: $LATEST_PRED"
                
                # Show file size
                PRED_SIZE=$(du -h "$LATEST_PRED" | cut -f1)
                echo "Prediction file size: $PRED_SIZE"
            else
                print_warning "Prediction file not found in results/predictions/"
            fi
        else
            print_error "Prediction generation failed"
            exit 1
        fi
    else
        print_warning "Skipping prediction generation (--skip-prediction)"
# Step 5: Create submission file
if should_run_step "5"; then
    if [ "$SKIP_SUBMISSION" = false ]; then
        print_header "STEP 5: Submission File Generation"
        
        # Find latest prediction
        LATEST_PRED=$(ls -t results/predictions/${MODEL_TYPE}_*.pkl 2>/dev/null | head -1)
        
        if [ -n "$LATEST_PRED" ]; then
            print_step "Creating Kaggle submission file..."
            print_step "Input: $LATEST_PRED"
            
            SUBMISSION_FILE="results/submissions/2020320135_${MODEL_TYPE}.csv"
            
            if python3 scripts/generate_submission.py \
                --predictions "$LATEST_PRED" \
                --output "$SUBMISSION_FILE" \
                --student_id "2020320135"; then
                print_success "Submission file created: $SUBMISSION_FILE"
                
                # Verify submission format
                SUBMISSION_LINES=$(wc -l < "$SUBMISSION_FILE")
                echo "Submission lines: $SUBMISSION_LINES (expected: 19659 = 1 header + 19658 test)"
                
                if [ $SUBMISSION_LINES -eq 19659 ]; then
                    print_success "Submission format verified ✓"
                else
                    print_warning "Unexpected line count. Expected 19659, got $SUBMISSION_LINES"
                fi
                
                # Show file size
                SUBM_SIZE=$(du -h "$SUBMISSION_FILE" | cut -f1)
                echo "Submission file size: $SUBM_SIZE"
            else
                print_error "Submission file generation failed"
                exit 1
            fi
        else
            print_warning "No predictions found. Run step 4 first or remove --skip-prediction."
        fi
    else
        print_warning "Skipping submission file generation (--skip-submission)"
    fi
if [ -z "$RUN_STEP" ]; then
    print_header "PIPELINE COMPLETED SUCCESSFULLY"

    echo -e "${GREEN}Summary:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Model Type:        $MODEL_TYPE"
    echo "Model Location:    data/models/$MODEL_TYPE/best_model.pt"
    echo "Training History:  data/models/$MODEL_TYPE/training_history.json"
    echo "Training Plots:    results/training/$MODEL_TYPE/"
    
    if [ -d "results/evaluation/$MODEL_TYPE" ] && [ "$SKIP_EVALUATION" = false ]; then
        echo "Evaluation Results: results/evaluation/$MODEL_TYPE/"
        if [ -f "results/evaluation/$MODEL_TYPE/evaluation_metrics.json" ]; then
            echo "Evaluation Metrics: results/evaluation/$MODEL_TYPE/evaluation_metrics.json"
        fi
    fi

    if [ -n "$LATEST_PRED" ]; then
        echo "Predictions:       $LATEST_PRED"
    fi

    if [ -f "$SUBMISSION_FILE" ]; then
        echo "Submission File:   $SUBMISSION_FILE"
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    echo -e "\n${GREEN}Data Usage:${NC}"
    echo "  Training:   29,487 (train) + 19,658 (test) = 49,145 samples"
    echo "  Method:     Train labeled + Test unlabeled with silver labels"
    echo "  Prediction: 19,658 test samples (no labels)"

    echo -e "\n${GREEN}Next Steps:${NC}"
    echo "1. Review training plots: ls results/training/$MODEL_TYPE/"
    if [ -d "results/evaluation/$MODEL_TYPE" ] && [ "$SKIP_EVALUATION" = false ]; then
        echo "2. Review evaluation plots: ls results/evaluation/$MODEL_TYPE/"
        echo "3. Check evaluation metrics: cat results/evaluation/$MODEL_TYPE/evaluation_metrics.json"
        echo "4. Analyze results: jupyter notebook notebooks/Ablation_Analysis.ipynb"
        echo "5. Submit to Kaggle: Upload $SUBMISSION_FILE"
    else
        echo "2. Check training history: cat data/models/$MODEL_TYPE/training_history.json"
        echo "3. Analyze results: jupyter notebook notebooks/Ablation_Analysis.ipynb"
        echo "4. Submit to Kaggle: Upload $SUBMISSION_FILE"
    fi

    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Pipeline execution completed at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}\n"
fi

echo -e "${GREEN}Summary:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Model Type:        $MODEL_TYPE"
echo "Model Location:    models/$MODEL_TYPE/best_model.pt"

if [ -n "$LATEST_PRED" ]; then
    echo "Predictions:       $LATEST_PRED"
fi

if [ -f "$SUBMISSION_FILE" ]; then
    echo "Submission:        $SUBMISSION_FILE"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -e "\n${GREEN}Next Steps:${NC}"
echo "1. Review training history: cat models/$MODEL_TYPE/training_history.json"
echo "2. Analyze results: jupyter notebook notebooks/Ablation_Analysis.ipynb"
echo "3. Submit to Kaggle: Upload $SUBMISSION_FILE"

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Pipeline execution completed at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}\n"

exit 0
