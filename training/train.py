"""
Train MultiTask-CodePilot using QLoRA.
"""

from trl import SFTTrainer, SFTConfig
from training.config import NUM_WORKERS

from training.config import (
    DATASET_PATH,
    OUTPUT_DIR,
    NUM_EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    LOGGING_STEPS,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    FP16,
    BF16,
    SEED,
    REPORT_TO,
)

from training.dataset import load_training_dataset
from training.model import load_model


def main():

    print("=" * 60)
    print("MultiTask-CodePilot Training")
    print("=" * 60)

    # --------------------------------------------------
    # Load model & tokenizer
    # --------------------------------------------------

    model, tokenizer = load_model()

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    dataset = load_training_dataset(DATASET_PATH)

    print(f"Dataset Size : {len(dataset):,}")

    # --------------------------------------------------
    # Training Arguments
    # --------------------------------------------------

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,

        logging_steps=LOGGING_STEPS,

        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,

        fp16=False,
        bf16=False,

        optim="adamw_torch",

        lr_scheduler_type="cosine",

        report_to=REPORT_TO,

        seed=SEED,

        remove_unused_columns=False,

        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=True,
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    print("\nStarting Training...\n")

    trainer.train()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    print("\nSaving LoRA Adapter...")

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()