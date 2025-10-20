"""
traffic_signs_train.py

Usage:
    python traffic_signs_train.py --data_dir ./GTSRB --img_size 128 --batch_size 32 --epochs 20 --fine_tune_epochs 10

Requires:
    pip install tensorflow opencv-python matplotlib scikit-learn numpy pandas

Notes:
 - Expects dataset prepared as:
    GTSRB/
      train/
        0/
        1/
        ...
      test/    (optional)
        0/
        1/
        ...
 - This script will:
    1) Create data generators with augmentation
    2) Build a ResNet50-based model (transfer learning)
    3) Train with frozen backbone, then fine-tune
    4) Save best model, final model and class mapping
    5) Plot training history and confusion matrix (if test/ present)
"""

import os
import argparse
import json
import math
import numpy as np 
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.applications import ResNet50 
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

def build_model_resnet(input_shape, n_classes, dropout_rate=0.5):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D(name='gap')(base.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inputs=base.input, outputs=outputs)
    return model, base

def plot_history(history, out_path=None):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val loss')
    plt.legend(); plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train acc')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend(); plt.title('Accuracy')

    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=6)
    plt.yticks(tick_marks, classes, fontsize=6)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize=5,
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def main(args):
    # paths
    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test") if os.path.isdir(os.path.join(data_dir, "test")) else None

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    img_size = (args.img_size, args.img_size)
    input_shape = (args.img_size, args.img_size, 3)
    batch_size = args.batch_size

    # ImageDataGenerators
    train_datagen = ImageDataGenerator(
        preprocessing_function=resnet_preprocess,
        rotation_range=15,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.12,
        zoom_range=0.12,
        horizontal_flip=False,
        validation_split=0.15  # use a portion of train for validation
    )

    test_datagen = ImageDataGenerator(preprocessing_function=resnet_preprocess)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    labels_map = train_gen.class_indices
    labels_inv = {v:k for k,v in labels_map.items()}
    n_classes = len(labels_map)
    print(f"Found {n_classes} classes")

    if test_dir:
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=1,
            class_mode='categorical',
            shuffle=False
        )
    else:
        test_gen = None

    # compute class weights
    # gather class indices from the training generator (it's shuffled by batch; access filenames)
    train_classes = []
    # We can use train_gen.classes property (works for DirectoryIterator)
    try:
        train_classes = train_gen.classes
    except Exception:
        # fallback: iterate through filenames
        for _, y in train_gen:
            train_classes.extend(np.argmax(y, axis=1))
            if len(train_classes) >= train_gen.samples:
                break
        train_classes = np.array(train_classes[:train_gen.samples])

    class_weights = class_weight.compute_class_weight('balanced', classes=np.arange(n_classes), y=train_classes)
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    print("Computed class weights (sample):", dict(list(class_weights_dict.items())[:5]))

    # Build model
    model, base = build_model_resnet(input_shape, n_classes, dropout_rate=args.dropout)
    model.summary()

    # Callbacks
    best_model_path = args.save_model or "best_traffic_resnet.h5"
    checkpoint = callbacks.ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early = callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    logdir = args.log_dir or "logs"
    tb = callbacks.TensorBoard(log_dir=logdir)

    # Phase 1: freeze backbone, train top
    for layer in base.layers:
        layer.trainable = False

    model.compile(optimizer=optimizers.Adam(learning_rate=args.lr), loss='categorical_crossentropy', metrics=['accuracy'])

    steps_per_epoch = math.ceil(train_gen.samples / batch_size)
    validation_steps = math.ceil(val_gen.samples / batch_size)

    print(f"Training top layers for {args.epochs} epochs (backbone frozen)...")
    history1 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        class_weight=class_weights_dict,
        callbacks=[checkpoint, early, reduce_lr, tb]
    )

    # Phase 2: fine-tune - unfreeze some of the top ResNet blocks
    if args.fine_tune_epochs > 0:
        # Unfreeze from a specific layer (tune this)
        # We'll unfreeze the last N layers (or layers after a certain name)
        unfreeze_at = args.unfreeze_at  # index or name
        print("Unfreeze strategy:", unfreeze_at)
        if isinstance(unfreeze_at, int) and unfreeze_at >= 0:
            for layer in base.layers[:unfreeze_at]:
                layer.trainable = False
            for layer in base.layers[unfreeze_at:]:
                layer.trainable = True
        else:
            # default: unfreeze last 30 layers
            for layer in base.layers[:-30]:
                layer.trainable = False
            for layer in base.layers[-30:]:
                layer.trainable = True

        # recompile with lower lr
        model.compile(optimizer=optimizers.Adam(learning_rate=args.fine_tune_lr),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Fine-tuning for {args.fine_tune_epochs} epochs (some backbone layers trainable)...")
        history2 = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs + args.fine_tune_epochs,
            initial_epoch=args.epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            class_weight=class_weights_dict,
            callbacks=[checkpoint, early, reduce_lr, tb]
        )
        # combine histories
        # create final history object similar to Keras History
        history = history1
        # append history2
        for k, v in history2.history.items():
            history.history.setdefault(k, []).extend(v)
    else:
        history = history1

    # Save final model and mapping
    final_model_path = args.final_model or "traffic_resnet_final.h5"
    model.save(final_model_path)
    print("Saved final model to", final_model_path)

    mapping_path = args.class_map or "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(labels_map, f, indent=2)
    print("Saved class mapping to", mapping_path)

    # Plot training history
    plot_history(history, out_path="training_history.png")

    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(val_gen, verbose=1)
    print(f"Validation loss: {val_loss:.4f}, acc: {val_acc:.4f}")

    # If test set present: predict & confusion matrix
    if test_gen is not None:
        print("Evaluating on test set...")
        test_gen.reset()
        preds = model.predict(test_gen, verbose=1)
        y_pred = np.argmax(preds, axis=1)
        y_true = test_gen.classes
        cm = confusion_matrix(y_true, y_pred)
        # use labels names sorted by index
        label_names = [labels_inv[i] for i in range(n_classes)]
        plot_confusion_matrix(cm, classes=label_names, normalize=False, title='Confusion matrix - test')
        plot_confusion_matrix(cm, classes=label_names, normalize=True, title='Normalized confusion matrix - test')
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=label_names))
    else:
        print("No test/ directory found; skipped test evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Traffic Sign Recognition using ResNet50 transfer learning")
    parser.add_argument('--data_dir', type=str, default='./GTSRB', help='Path to GTSRB folder with train/ and optional test/')
    parser.add_argument('--img_size', type=int, default=128, help='Square image size - e.g., 64, 96, 128')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10, help='Initial epochs with backbone frozen')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Extra epochs for fine-tuning (unfreeze top layers)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for initial training')
    parser.add_argument('--fine_tune_lr', type=float, default=1e-4, help='Lower LR for fine-tuning')
    parser.add_argument('--unfreeze_at', type=int, default=-1,
                        help='If >=0, unfreeze base.layers[unfreeze_at:] else default last 30 layers')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--save_model', type=str, default='best_traffic_resnet.h5')
    parser.add_argument('--final_model', type=str, default='traffic_resnet_final.h5')
    parser.add_argument('--class_map', type=str, default='class_mapping.json')
    parser.add_argument('--log_dir', type=str, default='logs')
    args = parser.parse_args()
    main(args)
