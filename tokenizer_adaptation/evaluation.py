from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
from datasets import load_dataset

checkpoint = "FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForMaskedLM.from_pretrained(checkpoint)
dataset = load_dataset("wikimedia/wikipedia", "20231101.de", streaming=True)

training_args = TrainingArguments(
    output_dir="my_awesome_eli5_mlm_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)
trainer.train()