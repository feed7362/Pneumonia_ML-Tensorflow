import tensorflow as tf
import os

# Configuration
file_path = r"data.tfrecord"  # Replace with your TFRecord file
output_dir = r"./"  # Replace with your output directory
os.makedirs(output_dir, exist_ok=True)

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Ensure proportions sum to 1
assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."

# Count total records in the dataset
total_items = sum(1 for _ in tf.data.TFRecordDataset(file_path))
print(f"Total records: {total_items}")

train_size = int(total_items * train_ratio)
val_size = int(total_items * val_ratio)
test_size = total_items - train_size - val_size  # Ensure all items are accounted for

# Split datasets using batching
dataset = tf.data.TFRecordDataset(file_path).shuffle(buffer_size=total_items, seed=2)
train_dataset = dataset.take(train_size)
remaining_dataset = dataset.skip(train_size)
val_dataset = remaining_dataset.take(val_size)
test_dataset = remaining_dataset.skip(val_size)

# Save datasets
def write_tfrecord(dataset, file_path):
    writer = tf.io.TFRecordWriter(file_path)
    for record in dataset:
        try:
            writer.write(record.numpy())
        except Exception as e:
            print(f"Error writing record: {e}")
    writer.close()
    print(f"Saved: {file_path}")


write_tfrecord(train_dataset, os.path.join(output_dir, "train-gpu.tfrecord"))
write_tfrecord(val_dataset, os.path.join(output_dir, "val.tfrecord"))
write_tfrecord(test_dataset, os.path.join(output_dir, "test.tfrecord"))
print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")