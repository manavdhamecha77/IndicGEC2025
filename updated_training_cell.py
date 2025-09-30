# Updated IndicBART Training Cell - Replace in indicBART.ipynb
# This cell should replace the problematic training cells in your notebook

import torch
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
import pandas as pd
import numpy as np
from pathlib import Path

print("🚀 Starting IMPROVED IndicBART Training")
print("=" * 50)

# Load the new large dataset
print("📊 Loading new large dataset...")
train_path = Path('Hindi/train.csv')
dev_path = Path('Hindi/dev.csv')

if not train_path.exists() or not dev_path.exists():
    print("⚠️  New dataset files not found!")
    print("Please run the integrate_new_dataset.ipynb notebook first.")
    print("This will create Hindi/train.csv and Hindi/dev.csv with 10k+ samples")
    raise FileNotFoundError("Missing train/dev CSV files")

train_df = pd.read_csv(train_path)
dev_df = pd.read_csv(dev_path)

print(f"✅ Dataset loaded:")
print(f"   📄 Train: {len(train_df):,} samples")
print(f"   📄 Dev: {len(dev_df):,} samples")
print(f"   📊 Total: {len(train_df) + len(dev_df):,} samples")

# Show data composition
train_corrections = (train_df['Input sentence'] != train_df['Output sentence']).sum()
dev_corrections = (dev_df['Input sentence'] != dev_df['Output sentence']).sum()
print(f"   🔧 Error corrections: {train_corrections + dev_corrections:,}")
print(f"   🎯 Identity pairs: {len(train_df) + len(dev_df) - train_corrections - dev_corrections:,}")

# IMPROVED Configuration (fixes all previous issues)
MODEL_NAME = 'ai4bharat/IndicBART'
OUTPUT_DIR = './indicbart-hindi-improved'
MAX_LENGTH = 128  # Optimal for Hindi sentences

print(f"\n⚙️ Improved Configuration:")
print(f"   🔧 Lower learning rate: 5e-6 (was 1e-5)")
print(f"   🔧 Better generation params: repetition_penalty=1.5")
print(f"   🔧 Regularization: weight_decay=0.01")
print(f"   🔧 Early stopping: patience=2")
print(f"   🔧 Optimized batch size: 2 (with grad accumulation 8)")

# Load model and tokenizer fresh
print(f"\n🔄 Loading fresh model and tokenizer...")
torch.cuda.empty_cache()  # Clear GPU memory

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Fix tokenizer issues
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

print(f"   ✅ Model loaded: {type(model).__name__}")
print(f"   ✅ Tokenizer: {type(tokenizer).__name__}")
print(f"   ✅ Vocab size: {len(tokenizer):,}")

# IMPROVED Tokenization (simple and effective)
def tokenize_function(examples):
    """Simple tokenization with task prefix"""
    # Clean prompting - just add "सुधारें: " (means "correct:")
    inputs = [f"सुधारें: {text}" for text in examples['Input sentence']]
    targets = examples['Output sentence']
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False  # Dynamic padding later
    )
    
    # Tokenize targets
    labels = tokenizer(
        targets,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Create and tokenize datasets
print(f"\n🔄 Creating datasets...")
train_dataset = Dataset.from_pandas(train_df)
dev_dataset = Dataset.from_pandas(dev_df)

print(f"   🔄 Tokenizing datasets...")
train_tokenized = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train"
)

dev_tokenized = dev_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dev_dataset.column_names,
    desc="Tokenizing dev"
)

print(f"   ✅ Tokenization complete")
print(f"   📊 Train tokens: {len(train_tokenized)}")
print(f"   📊 Dev tokens: {len(dev_tokenized)}")

# Data collator for dynamic padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# IMPROVED Training Arguments (fixes overfitting and instability)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Training schedule
    num_train_epochs=5,              # Fewer epochs to prevent overfitting
    per_device_train_batch_size=2,   # Manageable batch size
    per_device_eval_batch_size=4,    # Larger eval batches
    gradient_accumulation_steps=8,   # Effective batch size = 2*8 = 16
    
    # Learning rates and optimization
    learning_rate=5e-6,              # Lower learning rate for stability
    warmup_ratio=0.1,                # Gradual warmup
    weight_decay=0.01,               # Regularization
    max_grad_norm=1.0,               # Gradient clipping
    
    # Evaluation and saving
    evaluation_strategy="steps",     # Regular evaluation
    eval_steps=500,                  # Evaluate every 500 steps
    save_steps=500,                  # Save every 500 steps
    save_total_limit=3,              # Keep only 3 checkpoints
    load_best_model_at_end=True,     # Load best model
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Logging and optimization
    logging_steps=100,               # Log frequently
    fp16=torch.cuda.is_available(),  # Use mixed precision if available
    dataloader_pin_memory=False,     # Prevent memory issues
    remove_unused_columns=False,     # Keep all columns
    report_to=None,                  # No wandb/tensorboard
    seed=42,                         # Reproducibility
    
    # Performance optimizations
    dataloader_num_workers=0,        # Prevent multiprocessing issues
    greater_is_better=False,
    prediction_loss_only=False,
)

print(f"\n⚙️ Training configuration:")
print(f"   📊 Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   📈 Learning rate: {training_args.learning_rate}")
print(f"   🔄 Epochs: {training_args.num_train_epochs}")
print(f"   💾 Output dir: {OUTPUT_DIR}")

# Create trainer with early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=dev_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2)  # Stop if no improvement
    ]
)

print(f"\n🎯 Trainer created with {len(train_tokenized):,} training samples")

# Start improved training
print(f"\n🚀 Starting improved training...")
print(f"   ⏱️ Estimated time: ~2-3 hours")
print(f"   💾 Checkpoints will be saved every 500 steps")
print(f"   📊 Evaluation every 500 steps")

try:
    # Train the model
    training_output = trainer.train()
    
    print(f"\n✅ Training completed successfully!")
    print(f"   📉 Final train loss: {training_output.training_loss:.4f}")
    
    # Save final model
    print(f"\n💾 Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"   ✅ Model saved to: {OUTPUT_DIR}")
    
    # Final evaluation
    print(f"\n📊 Final evaluation...")
    eval_results = trainer.evaluate()
    print(f"   📉 Final eval loss: {eval_results['eval_loss']:.4f}")
    
    # Store results
    globals()['trained_model'] = model
    globals()['trained_tokenizer'] = tokenizer
    globals()['training_completed'] = True
    globals()['training_results'] = training_output
    
    print(f"\n🎉 TRAINING SUCCESS!")
    
except Exception as e:
    print(f"\n❌ Training failed: {str(e)}")
    print(f"💡 Check GPU memory and try reducing batch size if needed")
    raise

# IMPROVED Test Function (fixes repetition issues)
def test_grammar_correction(text, max_length=128):
    """Test grammar correction with improved generation parameters"""
    # Clean input formatting
    input_text = f"सुधारें: {text.strip()}"
    
    # Tokenize
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=max_length, 
        truncation=True
    )
    
    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate with IMPROVED parameters (fixes repetition)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            
            # Length control
            max_new_tokens=50,           # Limit new tokens
            min_new_tokens=1,            # Ensure some output
            
            # Quality control
            num_beams=3,                 # Moderate beam search
            early_stopping=True,         # Stop when EOS generated
            
            # REPETITION FIXES
            repetition_penalty=1.5,      # Strong penalty for repetition
            no_repeat_ngram_size=3,      # Block 3-gram repetitions
            length_penalty=0.8,          # Prefer shorter outputs
            
            # Sampling (disabled for consistency)
            do_sample=False,             # Deterministic
            temperature=1.0,             # Not used with do_sample=False
            
            # Token handling
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
    
    # Decode result
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean output (remove prompt prefix)
    if result.startswith("सुधारें: "):
        result = result[8:].strip()
    
    # Additional cleaning
    result = result.strip()
    
    # Return original if empty or too similar to input
    if not result or result == text.strip():
        return text.strip()
    
    return result

# Test the improved model
print(f"\n🧪 Testing improved model with fixed generation:")
test_sentences = [
    "मैं कल दिल्ली जाऊगा",      # Missing anusvara (should be जाऊंगा)
    "वो स्कूल गया हैं",         # Verb agreement (should be गया है)
    "राम और श्याम खेल रहा है",   # Plural subject (should be खेल रहे हैं)
    "मुझे यह किताब पसंद हैं",    # Agreement error (should be पसंद है)
    "बच्चे पार्क में खेल रहे हैं"  # Correct sentence (should stay same)
]

print(f"\n📋 Test Results:")
for i, sentence in enumerate(test_sentences, 1):
    try:
        corrected = test_grammar_correction(sentence)
        status = "✅ Changed" if corrected != sentence else "⚪ Unchanged"
        
        print(f"\n{i}. {status}")
        print(f"   Original:  {sentence}")
        print(f"   Corrected: {corrected}")
        
    except Exception as e:
        print(f"\n{i}. ❌ Error: {str(e)[:50]}...")

print(f"\n🎯 Key Improvements Made:")
print(f"   🔢 Dataset: {len(train_df):,} samples (vs ~600 before)")
print(f"   🛠️ Generation: Fixed repetition with penalties")
print(f"   ⚙️ Training: Lower LR, regularization, early stopping")
print(f"   💾 Checkpointing: Best model saved automatically")
print(f"   🎯 Focus: Grammar correction, not repetitive loops")

print(f"\n✅ Ready for evaluation and testing!")
print(f"💡 Try: test_grammar_correction('your sentence here')")
