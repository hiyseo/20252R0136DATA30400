#!/bin/bash

################################################################################
# run.sh - Complete Pipeline Execution Script
# Author: 20252R0136
# Description: Executes the entire hierarchical classification pipeline
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
SKIP_PREDICTION=false
MODEL_TYPE="baseline"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-labels)
            SKIP_LABELS=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-prediction)
            SKIP_PREDICTION=true
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
            echo "  --skip-labels       Skip silver label generation"
            echo "  --skip-training     Skip model training"
            echo "  --skip-prediction   Skip prediction generation"
            echo "  --model-type TYPE   Model type (default: baseline)"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run.sh                           # Run full pipeline"
            echo "  ./run.sh --skip-labels             # Skip label generation"
            echo "  ./run.sh --model-type focal_loss   # Use focal loss model"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

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
    print_success "Data preprocessing completed"
else
    print_error "Data preprocessing failed"
    exit 1
fi

# Step 2: Silver label generation
if [ "$SKIP_LABELS" = false ]; then
    print_header "STEP 2: Silver Label Generation"
    print_step "Generating silver labels with hybrid top-down approach..."
    print_step "Method: 0.3×keyword + 0.7×semantic + hierarchy filtering"
    
    if python3 scripts/generate_labels.py; then
        print_success "Silver labels generated successfully"
        
        # Check generated files
        if [ -f "data/intermediate/train_silver_labels.pkl" ] && [ -f "data/intermediate/test_silver_labels.pkl" ]; then
            print_success "Label files verified"
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
    fi
fi

# Step 3: Model training
if [ "$SKIP_TRAINING" = false ]; then
    print_header "STEP 3: Model Training (2-Stage)"
    print_step "Stage 1: BCE initialization (2 epochs)"
    print_step "Stage 2: Self-training with KLD (3 iterations)"
    print_step "Model type: $MODEL_TYPE"
    
    if python3 scripts/train_with_config.py; then
        print_success "Training completed successfully"
        
        # Check model file
        if [ -f "models/$MODEL_TYPE/best_model.pt" ]; then
            print_success "Model saved: models/$MODEL_TYPE/best_model.pt"
            
            # Show model size
            MODEL_SIZE=$(du -h "models/$MODEL_TYPE/best_model.pt" | cut -f1)
            echo "Model size: $MODEL_SIZE"
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
    if [ ! -f "models/$MODEL_TYPE/best_model.pt" ]; then
        print_error "Model not found: models/$MODEL_TYPE/best_model.pt"
        print_error "Remove --skip-training to train the model."
        exit 1
    fi
fi

# Step 4: Prediction generation
if [ "$SKIP_PREDICTION" = false ]; then
    print_header "STEP 4: Prediction Generation"
    print_step "Generating predictions on test set..."
    
    if python3 src/inference/predict.py \
        --model_path "models/$MODEL_TYPE/best_model.pt" \
        --model_name "$MODEL_TYPE"; then
        print_success "Predictions generated successfully"
        
        # Find latest prediction file
        LATEST_PRED=$(ls -t results/predictions/${MODEL_TYPE}_*.csv 2>/dev/null | head -1)
        if [ -n "$LATEST_PRED" ]; then
            print_success "Predictions saved: $LATEST_PRED"
            
            # Count predictions
            PRED_COUNT=$(($(wc -l < "$LATEST_PRED") - 1))
            echo "Prediction count: $PRED_COUNT"
        else
            print_warning "Prediction file not found in results/predictions/"
        fi
    else
        print_error "Prediction generation failed"
        exit 1
    fi
else
    print_warning "Skipping prediction generation (--skip-prediction)"
fi

# Step 5: Create submission file
print_header "STEP 5: Submission File Generation"

# Find latest prediction
LATEST_PRED=$(ls -t results/predictions/${MODEL_TYPE}_*.csv 2>/dev/null | head -1)

if [ -n "$LATEST_PRED" ]; then
    print_step "Creating Kaggle submission file..."
    
    SUBMISSION_FILE="results/submissions/20252R0136_${MODEL_TYPE}.csv"
    
    if python3 scripts/generate_submission.py \
        --predictions "$LATEST_PRED" \
        --output "$SUBMISSION_FILE" \
        --student_id "20252R0136"; then
        print_success "Submission file created: $SUBMISSION_FILE"
        
        # Verify submission format
        SUBMISSION_LINES=$(wc -l < "$SUBMISSION_FILE")
        echo "Submission lines: $SUBMISSION_LINES (expected: 19659)"
        
        if [ $SUBMISSION_LINES -eq 19659 ]; then
            print_success "Submission format verified ✓"
        else
            print_warning "Unexpected line count. Expected 19659, got $SUBMISSION_LINES"
        fi
    else
        print_error "Submission file generation failed"
        exit 1
    fi
else
    print_warning "No predictions found. Skipping submission generation."
fi

################################################################################
# Summary
################################################################################

print_header "PIPELINE COMPLETED SUCCESSFULLY"

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
