"""Run an encoder-decoder model using Huggingface (if running on Google Colab)."""
# 1. Install HuggingFace libraries
# !pip install transformers datasets
# # 2. Load your custom dataset (CSV with input_text, target_text columns)
from datasets import load_dataset
from transformers import BertTokenizerFast, EncoderDecoderModel
import pandas as pd
dataset = load_dataset('csv', data_files={'train': '/content/train.csv', 'validation': '/content/test.csv'})

# 3. Load model and tokenizer
model_name = 'google/muril-base-cased'
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_pretrained_model_name_or_path=model_name, decoder_pretrained_model_name_or_path=model_name)
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
# 4. Tokenize data
def tokenize(batch):
    input_enc = tokenizer(batch['Input sentence'], max_length=64, truncation=True, padding='max_length')
    target_enc = tokenizer(batch['Output sentence'], max_length=64, truncation=True, padding='max_length')
    input_enc['labels'] = target_enc['input_ids']
    return input_enc

tokenized = dataset.map(tokenize, batched=True)

# 5. Trainer setup
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="results",
    num_train_epochs=5,
    logging_steps=25,
    logging_first_step=True,
    per_device_train_batch_size=2,
    eval_strategy="epoch"
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['validation'],
    tokenizer=tokenizer
)
trainer.train()
model.save_pretrained('final-encoder-decoder-model')
tokenizer.save_pretrained('final-encoder-decoder-model')
# 6. Inference
# If you want to load later (save this into your local system)
# tokenizer = BertTokenizerFast.from_pretrained('/content/final-encoder-decoder-model')
# model = EncoderDecoderModel.from_pretrained('/content/final-encoder-decoder-model')
test_string = "हम सुनते हैं कि ग्लेशियर पिघलने से जलस्तर में बढ़ोतरी हुई है"
input_tokens = tokenizer(test_string, return_tensors="pt")
output_ids = model.generate(**input_tokens, decoder_start_token_id=model.config.decoder_start_token_id)
print(output_ids)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
