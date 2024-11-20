import os
from pathlib import Path
from zipfile import ZipFile
import tensorflow as tf
from src.cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    """Prepare and update the base model for transfer learning."""

    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initialize the base model preparation with configuration.

        Args:
            config (PrepareBaseModelConfig): Configuration for preparing the base model.
        """
        self.config = config
        self.model = None  # Placeholder for the base model

    def get_base_model(self) -> None:
        """
        Load the base model (VGG16) with specified parameters and save it.
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        print("Base model loaded successfully.")

        # Save the base model
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model: tf.keras.Model, classes: int, freeze_all: bool, 
                            freeze_till: int, learning_rate: float) -> tf.keras.Model:
        """
        Prepare the full model by adding custom layers on top of the base model.

        Args:
            model (tf.keras.Model): Base model.
            classes (int): Number of output classes.
            freeze_all (bool): Whether to freeze all layers of the base model.
            freeze_till (int): Freeze layers up to this index. Ignored if `freeze_all` is True.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
            tf.keras.Model: Updated model with custom layers.
        """
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:freeze_till]:
                layer.trainable = False

        # Add custom layers
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        # Create full model
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Compile the model
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        print("Full model prepared successfully.")
        return full_model

    def update_base_model(self) -> None:
        """
        Update the base model by adding custom layers and freezing specified layers.
        """
        if self.model is None:
            raise ValueError("Base model is not loaded. Call `get_base_model()` first.")

        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        # Save the updated base model
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model) -> None:
        """
        Save the given model to the specified path.

        Args:
            path (Path): Path to save the model.
            model (tf.keras.Model): Model to be saved.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save(path)
        print(f"Model saved at {path}")
