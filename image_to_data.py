import os
import cv2
from tensorflow import io, train

dimension = 224
mode = 'rgb'

def image_to_pixel_values(image_path, size=(dimension, dimension)):
    try:
        if mode == 'grayscale':
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        else:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load as BGR

        if img is None:
            raise ValueError(f"Unable to load image at {image_path}")

        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img_flat = img.flatten().astype('uint8')
        return img_flat
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def write_tfrecord(path_list, output_file, size=(dimension, dimension)):
    with io.TFRecordWriter(output_file) as writer:
        for path in path_list:
            print(f"Processing directory: {path}")
            
            image_files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            if not image_files:
                print(f"No images found in {path}")
                continue

            for image_file in image_files:
                label = 0 if "NORMAL" in path.upper() else 1
                image_path = os.path.join(path, image_file)
                print(f"Processing image: {image_path} : {label}")
                pixel_values = image_to_pixel_values(image_path, size)
                if pixel_values is not None:
                    feature = {
                        'label': train.Feature(int64_list=train.Int64List(value=[label])),
                        'image': train.Feature(bytes_list=train.BytesList(value=[pixel_values.tobytes()]))
                    }
                    example = train.Example(features=train.Features(feature=feature))
                    writer.write(example.SerializeToString())

        print("TFRecord file created successfully.")

if __name__ == "__main__":
    # List of image directories
    path_list = [
        r'D:/Projects/Pneumonia_ML-Tensorflow/chest_xray/val/NORMAL/',
        r'D:/Projects/Pneumonia_ML-Tensorflow/chest_xray/val/PNEUMONIA/',
        r'D:/Projects/Pneumonia_ML-Tensorflow/chest_xray/test/NORMAL/',
        r'D:/Projects/Pneumonia_ML-Tensorflow/chest_xray/test/PNEUMONIA/',
        r'D:/Projects/Pneumonia_ML-Tensorflow/chest_xray/train-gpu/NORMAL/',
        r'D:/Projects/Pneumonia_ML-Tensorflow/chest_xray/train-gpu/PNEUMONIA/'
    ]
    # Convert images to TFRecord
    write_tfrecord(path_list, output_file='data.tfrecord', size=(dimension, dimension))
