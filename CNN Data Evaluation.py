from __future__ import annotations

import os
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_ZIP_PATH = Path(r"D:\CVNLP\seg_train.zip")
TEST_ZIP_PATH = Path(r"D:\CVNLP\seg_test.zip")
PRED_ZIP_PATH = Path(r"D:\CVNLP\seg_pred.zip")
DATA_DIR = Path("/mnt/data/dataset")
IMAGE_SIZE = (228, 228)
BATCH_SIZE = 32
EPOCHS = 20


def extract_zip(file_path: Path, extract_to: Path) -> None:
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def count_images_in_classes(base_path: Path) -> dict[str, int]:
    class_counts: dict[str, int] = {}
    for category in sorted(base_path.iterdir()):
        if category.is_dir():
            class_counts[category.name] = sum(1 for item in category.iterdir() if item.is_file())
    return class_counts


def image_stats(image_folder: Path) -> Counter[tuple[int, int]]:
    dimensions: list[tuple[int, int]] = []
    for category in sorted(image_folder.iterdir()):
        if not category.is_dir():
            continue
        for img_path in category.iterdir():
            if not img_path.is_file():
                continue
            try:
                with Image.open(img_path) as img:
                    dimensions.append(img.size)
            except Exception as exc:
                print(f"Error loading image {img_path}: {exc}")
    return Counter(dimensions)


def visualize_samples(base_path: Path, num_samples: int = 5) -> None:
    categories = [item for item in sorted(base_path.iterdir()) if item.is_dir()][:num_samples]
    if not categories:
        print(f"No class folders found in {base_path}")
        return

    fig, axes = plt.subplots(1, len(categories), figsize=(15, 5))
    if len(categories) == 1:
        axes = [axes]

    for axis, category in zip(axes, categories):
        sample_image = next((item for item in sorted(category.iterdir()) if item.is_file()), None)
        if sample_image is None:
            axis.set_title(category.name)
            axis.axis("off")
            continue
        with Image.open(sample_image) as img:
            axis.imshow(img)
            axis.set_title(category.name)
            axis.axis("off")

    plt.tight_layout()
    plt.show()


def build_model(num_classes: int) -> Sequential:
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def create_generators(train_dir: Path, test_dir: Path):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    test_data = eval_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_data, test_data


def plot_history(history) -> None:
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.show()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.show()


def prepare_dataset() -> tuple[Path, Path, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    extract_zip(TRAIN_ZIP_PATH, DATA_DIR / "train")
    extract_zip(TEST_ZIP_PATH, DATA_DIR / "test")
    extract_zip(PRED_ZIP_PATH, DATA_DIR / "pred")

    train_dir = DATA_DIR / "train" / "seg_train"
    test_dir = DATA_DIR / "test" / "seg_test"
    pred_dir = DATA_DIR / "pred"
    return train_dir, test_dir, pred_dir


def run_prediction(model: Sequential, pred_dir: Path) -> None:
    if not pred_dir.exists():
        print("Prediction folder not found.")
        return

    pred_datagen = ImageDataGenerator(rescale=1.0 / 255)
    try:
        pred_data = pred_datagen.flow_from_directory(
            pred_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode=None,
            shuffle=False,
        )
    except Exception as exc:
        print(f"Prediction data could not be loaded from {pred_dir}: {exc}")
        return

    predictions = model.predict(pred_data)
    predicted_classes = tf.argmax(predictions, axis=-1)
    print("Predicted classes:", predicted_classes.numpy())


def main() -> None:
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

    train_dir, test_dir, pred_dir = prepare_dataset()

    print("Training set class distribution:")
    print(count_images_in_classes(train_dir))
    print("\nTraining set image dimensions:")
    print(image_stats(train_dir))

    print("\nTesting set class distribution:")
    print(count_images_in_classes(test_dir))
    print("\nTesting set image dimensions:")
    print(image_stats(test_dir))

    print("\nVisualizing training samples:")
    visualize_samples(train_dir)
    print("\nVisualizing testing samples:")
    visualize_samples(test_dir)

    train_data, test_data = create_generators(train_dir, test_dir)

    images, labels = next(train_data)
    print("Training batch shape:", images.shape)
    print("Training labels shape:", labels.shape)

    images, labels = next(test_data)
    print("Testing batch shape:", images.shape)
    print("Testing labels shape:", labels.shape)

    model = build_model(train_data.num_classes)
    model.summary()

    callbacks = [
        ModelCheckpoint(
            filepath="best_model.keras",
            monitor="val_accuracy",
            mode="auto",
            verbose=1,
            save_best_only=True,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="auto",
            min_delta=0.01,
            patience=5,
            verbose=1,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    loss, accuracy = model.evaluate(test_data)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    plot_history(history)
    run_prediction(model, pred_dir)


if __name__ == "__main__":
    main()
