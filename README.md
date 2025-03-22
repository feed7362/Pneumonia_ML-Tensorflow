# Pneumonia Detection using TensorFlow

## Overview
This repository contains a deep learning-based pneumonia detection system using TensorFlow. The goal is to classify chest X-ray images into pneumonia and non-pneumonia categories using convolutional neural networks (CNNs). The implementation includes optimizations for both CPU and GPU to maximize training efficiency.

## Dataset
- The dataset consists of labeled chest X-ray images.
- Samples:
    - **Training Set**: Contains a large number of labeled images for model training.
    - **Validation Set**: Used to fine-tune hyperparameters.
    - **Test Set**: Evaluates model performance on unseen data.
- **TFRecords Format**:
    - The dataset is preprocessed and stored in TFRecord format for efficient I/O operations and faster training.

## Model Architecture
- Uses **Convolutional Neural Networks (CNNs)** optimized for medical image classification.
- Implemented with **TensorFlow and Keras**.
- Features techniques like **batch normalization, dropout, and data augmentation** to enhance generalization.

## Unique Approaches
1. **TFRecords for Efficient Data Loading**:
    - TFRecords significantly speed up training by reducing disk I/O overhead.
    - Preprocessing is handled efficiently using `tf.data` pipelines.
2. **Separate CPU & GPU Implementations**:
    - The repository includes both CPU and GPU-specific training scripts.
    - The GPU version utilizes **TensorFlow's XLA compilation and mixed-precision training** for acceleration.
    - The CPU version is optimized with **Intel MKL-DNN** for better performance on non-GPU systems.
3. **Transfer Learning**:
    - Fine-tuned on pneumonia dataset to improve accuracy.
4. **Data Augmentation**:
    - Random transformations like **flipping, rotation, contrast adjustment** to improve model robustness.

## Training Process
### CPU vs. GPU Differences
| Feature         | CPU Version | GPU Version |
|---------------|------------|------------|
| Data Loading  | Standard pipeline | TFRecords-based pipeline |
| Processing Speed | Slower | Faster (optimized with mixed precision) |
| Model Compilation | Standard TensorFlow graph | XLA-compiled execution |
| Batch Size | Limited due to memory | Larger batch sizes supported |

## Usage
### Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib (for visualization)

### Running the Model
#### Train on CPU
```bash
python train_cpu.py
```
#### Train on GPU
```bash
python train_gpu.py
```
## Results & Performance
- Achieves high accuracy in pneumonia classification.
- GPU-accelerated training reduces time significantly compared to CPU execution.
- Fine-tuned CNNs outperform standard architectures in medical image classification.

## Future Enhancements
- Implement attention mechanisms for improved interpretability.
- Experiment with additional model architectures like Vision Transformers.
- Extend to multi-class classification for other lung diseases.

## Contributing
Feel free to submit issues or contribute improvements via pull requests.

## License
This project is licensed under the MIT License.

