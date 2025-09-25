# Training the Octo Model

This guide explains how to train the Octo language model from scratch or resume training from a checkpoint.

## Prerequisites

- Python 3.12
- PyTorch (with CUDA support for GPU training)
- Required Python packages:
  - `torch`
  - `transformers`
  - `datasets`
  - `tqdm`

Install dependencies using pip:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # For CUDA 12.1
pip install transformers datasets tqdm
```

## Quick Start

To train with the default configuration:

```bash
python train/train.py --config train/config.json
```

Or use the provided script:
```bash
./train.sh
```

## Configuration

Training is controlled by two configuration files:

### Training Configuration (`train/config.json`)

The training configuration contains hyperparameters and dataset settings:

```json
{
  "dataset_name": "roneneldan/TinyStories",
  "train_split": "train",
  "eval_split": "validation",
  "text_column": "text",
  "tokenizer_name": "gpt2",
  "block_size": 512,
  "batch_size": 16,
  "gradient_accumulation_steps": 4,
  "num_epochs": 1,
  "learning_rate": 0.0003,
  "weight_decay": 0.01,
  "warmup_steps": 100,
  "max_grad_norm": 1.0,
  "seed": 42,
  "log_interval_steps": 50,
  "eval_interval_steps": 500,
  "save_interval_steps": 5000,
  "eval_max_batches": 64,
  "checkpoint_dir": "checkpoints",
  "resume_from": null,
  "use_mixed_precision": true,
  "use_gradient_checkpointing": false
}
```

#### Key Parameters

- **Dataset Settings**:
  - `dataset_name`: HuggingFace dataset name (e.g., "roneneldan/TinyStories")
  - `train_split`/`eval_split`: Dataset splits for training/evaluation
  - `text_column`: Column name containing text data

- **Tokenization**:
  - `tokenizer_name`: HuggingFace tokenizer name (e.g., "gpt2")
  - `block_size`: Maximum sequence length for training

- **Training Hyperparameters**:
  - `batch_size`: Batch size per GPU
  - `gradient_accumulation_steps`: Accumulate gradients over multiple steps
  - `num_epochs`: Number of training epochs
  - `learning_rate`: Learning rate for AdamW optimizer
  - `weight_decay`: Weight decay coefficient
  - `warmup_steps`: Number of warmup steps for learning rate scheduler
  - `max_grad_norm`: Maximum gradient norm for clipping

- **Logging and Checkpointing**:
  - `log_interval_steps`: Log training metrics every N steps
  - `eval_interval_steps`: Run evaluation every N steps
  - `save_interval_steps`: Save checkpoint every N steps
  - `checkpoint_dir`: Directory to save checkpoints

- **Performance Options**:
  - `use_mixed_precision`: Enable FP16 mixed precision training
  - `use_gradient_checkpointing`: Trade compute for memory (useful for large models)

### Model Configuration

The model architecture can be customized by creating a JSON file with OctoConfig parameters and passing it via `--model-config`:

```bash
python train/train.py --config train/config.json --model-config my_model_config.json
```

Example model config:
```json
{
  "vocab_size": 16000,
  "hidden_size": 768,
  "intermediate_size": 3072,
  "num_hidden_layers": 12,
  "num_attention_heads": 12,
  "num_key_value_heads": 3,
  "max_position_embeddings": 4096,
  "rms_norm_eps": 1e-6,
  "rope_theta": 10000.0,
  "attention_bias": false
}
```

#### Model Parameters

- `vocab_size`: Size of the tokenizer vocabulary
- `hidden_size`: Hidden dimension size
- `intermediate_size`: MLP intermediate dimension
- `num_hidden_layers`: Number of transformer layers
- `num_attention_heads`: Number of attention heads
- `num_key_value_heads`: Number of key/value heads (for grouped attention)
- `max_position_embeddings`: Maximum sequence length the model can handle
- `rope_theta`: RoPE (Rotary Position Embedding) base frequency

## Training Process

### Data Loading

The training script uses HuggingFace datasets for efficient streaming data loading:

1. Downloads the specified dataset
2. Tokenizes text into fixed-size blocks
3. Creates streaming datasets that don't require loading the entire dataset into memory

### Training Loop

The training process includes:

1. **Initialization**: Load model, tokenizer, and data loaders
2. **Training**: Iterate through batches with:
   - Forward pass and loss computation
   - Gradient accumulation and optimization
   - Learning rate scheduling
3. **Evaluation**: Periodic validation on eval set
4. **Checkpointing**: Save model state at regular intervals
5. **Logging**: Track loss, perplexity, and memory usage

### Monitoring Progress

Training progress is logged to the console and `logs/` directory:

- **Loss**: Training loss (cross-entropy)
- **Perplexity**: exp(loss) - lower is better
- **Memory**: GPU memory usage (if enabled)
- **Step/Epoch**: Current training progress

Example log output:
```
Step 100: loss=8.1234 ppl=3378.45
Step 200: loss=7.4567 ppl=1723.89 mem=4.2GB/8GB
```

## Checkpointing and Resuming

### Saving Checkpoints

Checkpoints are automatically saved to the `checkpoint_dir` at regular intervals. Each checkpoint contains:

- Model state dict
- Optimizer state
- Scheduler state
- Training step count
- Random seed state

### Resuming Training

To resume from a checkpoint, set `resume_from` in the config:

```json
{
  "resume_from": "checkpoints/checkpoint_step_5000.pt"
}
```

Or resume from the latest checkpoint:
```json
{
  "resume_from": "checkpoints"
}
```

The script will automatically find the latest checkpoint in the directory.

## Memory Optimization

For training larger models or with limited GPU memory:

1. **Gradient Checkpointing**: Set `use_gradient_checkpointing: true`
2. **Gradient Accumulation**: Increase `gradient_accumulation_steps`
3. **Reduce Batch Size**: Lower `batch_size`
4. **Mixed Precision**: Keep `use_mixed_precision: true`
5. **CPU Offloading**: Not currently supported, consider model parallelism for very large models

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce `batch_size`
   - Increase `gradient_accumulation_steps`
   - Enable `use_gradient_checkpointing`

2. **Dataset download fails**:
   - Check internet connection
   - Verify dataset name exists on HuggingFace
   - Use local dataset files if available

3. **Tokenizer issues**:
   - Ensure `tokenizer_name` matches a valid HuggingFace tokenizer
   - Check that `vocab_size` matches tokenizer vocabulary

4. **Slow training**:
   - Ensure CUDA is available (`python check.py`)
   - Consider using mixed precision
   - Check data loading bottleneck (reduce `buffer_size` in data.py if needed)

### Performance Tips

- Use SSD storage for datasets and checkpoints
- Monitor GPU utilization with `nvidia-smi`
- Adjust logging frequency to reduce I/O overhead
- Use appropriate batch sizes for your GPU memory

## Advanced Usage

### Custom Datasets

To use a custom dataset:

1. Upload to HuggingFace Hub or prepare local files
2. Update `dataset_name` to your dataset path
3. Adjust `text_column` if different from "text"
4. Modify splits as needed

### Multi-GPU Training

Currently single-GPU training is supported. For multi-GPU:

- Use PyTorch DDP (DistributedDataParallel)
- Modify training script to wrap model with DDP
- Set appropriate environment variables

### Hyperparameter Tuning

Key hyperparameters to experiment with:

- Learning rate (typically 1e-4 to 5e-4)
- Batch size (larger = more stable gradients)
- Warmup steps (10-20% of total steps)
- Weight decay (0.01-0.1)
- Model size (hidden_size, num_layers)

## Output Files

After training completes:

- **Checkpoints**: `checkpoints/checkpoint_step_XXXX.pt`
- **Logs**: `logs/training_YYYYMMDD_HHMMSS.log`
- **Final Model**: Ready for inference or further fine-tuning

The trained model can be loaded using the OctoForCausalLM class for text generation tasks.
