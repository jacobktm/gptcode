import os
import re
import numpy as np
from pathlib import Path
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
import tensorflow as tf

# Set the memory limit (in bytes)
MEMORY_LIMIT = 6.5 * 1024  # 6.5 GB
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_LIMIT)])

# Functions to load code samples in chunks
def split_into_chunks(tokens, max_length):
    chunks = []
    current_chunk = []
    for token in tokens:
        if len(current_chunk) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = [token]
        else:
            current_chunk.append(token)
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def load_code_samples(root_dir, extensions, chunk_size, tokenizer, max_length):
    code_samples = []
    current_size = 0
    byte_limit = tokenizer.model_max_length * 4  # 4 bytes per token is a rough estimate

    for ext in extensions:
        for code_file_path in Path(root_dir).rglob(f'*.{ext}'):
            if not code_file_path.is_dir():
                with open(code_file_path, 'r', encoding='utf-8', errors='ignore') as code_file:
                    code_text = code_file.read()
                    preprocessed_code = preprocess_code(code_text)
                    tokens = tokenizer.encode(preprocessed_code, max_length=max_length, truncation=True)

                    # Break tokens into chunks based on max_length
                    token_chunks = split_into_chunks(tokens, max_length)

                    # Calculate the size of the chunks and check if the chunk_size is exceeded
                    decoded_chunks = [tokenizer.decode(chunk) for chunk in token_chunks]
                    chunk_sizes_in_bytes = [len(chunk.encode('utf-8')) for chunk in decoded_chunks]

                    for i, decoded_chunk in enumerate(decoded_chunks):
                        if current_size + chunk_sizes_in_bytes[i] > chunk_size:
                            yield code_samples
                            code_samples = []
                            current_size = 0

                        code_samples.append(decoded_chunk)
                        current_size += chunk_sizes_in_bytes[i]

    if code_samples:
        yield code_samples


def preprocess_code(code):
    # Remove unnecessary white spaces
    code = re.sub(r'\s+', ' ', code)
    return code

def create_dataset(tokenizer, chunks, max_length, stride):
    input_ids = []
    attention_masks = []

    for chunk in chunks:
        tokens = tokenizer.encode(chunk, max_length=max_length, truncation=True)

        for i in range(0, len(tokens), stride):
            input_ids.append(tokens[i:i + max_length])

    for ids in input_ids:
        attention_masks.append([1] * len(ids))

    input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=max_length, padding="post")
    attention_masks = tf.keras.preprocessing.sequence.pad_sequences(attention_masks, maxlen=max_length, padding="post")

    dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks))

    return dataset


# Parameters
ROOT_DIR = "/app/training"
CHUNK_SIZE = 512 * 1024 * 1024  # 100 MB, for example
MAX_LENGTH = 1024
STRIDE = 512
MODEL_NAME = "gpt2"

# Set up the tokenizer and model
my_tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
my_model = TFGPT2LMHeadModel.from_pretrained(MODEL_NAME)

# Set up the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=my_tokenizer, mlm=False
)

# Set up the training configuration
num_epochs = 3
batch_size = 1
save_steps = 10_000

# Extensions to search for in the repositories
EXTENSIONS = ["c", "cpp", "h", "py", "rs", "js", "html", "css", "sh", "asm", "h", "hpp", "s", "S"]

# Load and train on code samples in chunks
trained_model_chunks = []
for chunk_idx, code_samples_chunk in enumerate(load_code_samples(ROOT_DIR, EXTENSIONS, CHUNK_SIZE, my_tokenizer, MAX_LENGTH)):
    # Create the dataset for the current chunk
    dataset_chunk = create_dataset(my_tokenizer, code_samples_chunk, MAX_LENGTH, STRIDE)

    # Train the model on the current chunk
    my_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    my_model.fit(dataset_chunk.batch(batch_size), epochs=num_epochs)

    # Save the trained model
    model_save_dir = f"trained_model_chunk_{chunk_idx}"
    my_model.save_pretrained(model_save_dir)
    trained_model_chunks.append(model_save_dir)

# Function to average model weights
def average_model_weights(model_save_dirs):
    num_models = len(model_save_dirs)
    model_weights = [TFGPT2LMHeadModel.from_pretrained(save_dir).get_weights() for save_dir in model_save_dirs]

    # Compute the average weights
    avg_weights = [np.mean([weights[i] for weights in model_weights], axis=0) for i in range(len(model_weights[0]))]

    return avg_weights

# Compute the average weights
avg_weights = average_model_weights(trained_model_chunks)

# Create a new model and set its weights to the average weights
merged_model = TFGPT2LMHeadModel.from_pretrained(MODEL_NAME)
merged_model.set_weights(avg_weights)

# Save the merged model
merged_model.save_pretrained("merged_model")
