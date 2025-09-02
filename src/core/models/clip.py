"""
CLIP model implementation for vision-language understanding in H200 Intelligent Mug Positioning System.

This module provides CLIP integration for embedding generation and
text-image similarity computation for advanced mug positioning.
"""

# Standard library imports
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import clip
import numpy as np
import structlog
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# First-party imports
from src.core.models.base import BaseModel, ModelConfig, ModelState

# Initialize structured logger
logger = structlog.get_logger(__name__)


class CLIPVisionModel(BaseModel):
    """
    CLIP model for vision-language understanding with mug positioning focus.

    Provides text-image similarity computation, embedding generation,
    and prompt engineering for optimal mug placement.
    """

    # Available CLIP models
    AVAILABLE_MODELS = {
        "RN50": {"embedding_dim": 1024, "input_size": 224},
        "RN101": {"embedding_dim": 512, "input_size": 224},
        "RN50x4": {"embedding_dim": 640, "input_size": 288},
        "RN50x16": {"embedding_dim": 768, "input_size": 384},
        "RN50x64": {"embedding_dim": 1024, "input_size": 448},
        "ViT-B/32": {"embedding_dim": 512, "input_size": 224},
        "ViT-B/16": {"embedding_dim": 512, "input_size": 224},
        "ViT-L/14": {"embedding_dim": 768, "input_size": 224},
        "ViT-L/14@336px": {"embedding_dim": 768, "input_size": 336},
    }

    # Mug positioning prompts
    MUG_PROMPTS = [
        "a mug placed on a clean surface",
        "a coffee cup in optimal position",
        "a mug with good spacing from other objects",
        "a well-positioned drinking vessel",
        "a cup placed safely on the table",
        "a mug in an ergonomic position",
        "a beverage container with clear surroundings",
        "a mug positioned for easy access",
    ]

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        use_huggingface: bool = False,
        normalize_embeddings: bool = True,
        cache: Optional[Any] = None,
    ):
        """
        Initialize CLIP model.

        Args:
            model_name: CLIP model name
            use_huggingface: Use HuggingFace transformers instead of OpenAI CLIP
            normalize_embeddings: Normalize embeddings to unit vectors
            cache: Optional cache instance
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model: {model_name}. Must be one of {list(self.AVAILABLE_MODELS.keys())}"
            )

        # Create model config
        config = ModelConfig(
            name=f"clip_{model_name.replace('/', '_')}",
            version="1.0",
            model_type="vision_language",
            r2_key=f"models/clip/{model_name.replace('/', '_')}_v1.0.pt",
            max_batch_size=32,
            warmup_iterations=3,
            metadata={
                "model_name": model_name,
                "embedding_dim": self.AVAILABLE_MODELS[model_name]["embedding_dim"],
                "input_size": self.AVAILABLE_MODELS[model_name]["input_size"],
                "use_huggingface": use_huggingface,
                "normalize_embeddings": normalize_embeddings,
            },
        )

        super().__init__(config, cache)

        self.model_name = model_name
        self.use_huggingface = use_huggingface
        self.normalize_embeddings = normalize_embeddings
        self.embedding_dim = self.AVAILABLE_MODELS[model_name]["embedding_dim"]
        self.input_size = self.AVAILABLE_MODELS[model_name]["input_size"]

        # Model components
        self.clip_model = None
        self.preprocess = None
        self.tokenizer = None

        # Precomputed text embeddings cache
        self._text_embeddings_cache: Dict[str, torch.Tensor] = {}

    async def _load_model_impl(self) -> torch.nn.Module:
        """Load CLIP model."""
        if self.use_huggingface:
            # Use HuggingFace implementation
            model_id = f"openai/clip-{self.model_name.lower()}"
            self.clip_model = CLIPModel.from_pretrained(model_id)
            self.processor = CLIPProcessor.from_pretrained(model_id)
            model = self.clip_model
        else:
            # Use OpenAI CLIP
            model, preprocess = clip.load(self.model_name, device=self.config.device)
            self.clip_model = model
            self.preprocess = preprocess

        # Precompute mug prompt embeddings
        await self._precompute_prompt_embeddings()

        logger.info(
            "CLIP model loaded",
            model_name=self.model_name,
            embedding_dim=self.embedding_dim,
            use_huggingface=self.use_huggingface,
        )

        return model

    async def _precompute_prompt_embeddings(self) -> None:
        """Precompute embeddings for common prompts."""
        logger.info("Precomputing prompt embeddings...")

        for prompt in self.MUG_PROMPTS:
            embedding = await self._encode_text(prompt)
            self._text_embeddings_cache[prompt] = embedding

        logger.info(f"Precomputed {len(self.MUG_PROMPTS)} prompt embeddings")

    async def preprocess(
        self, inputs: Union[Image.Image, List[Image.Image], str, List[str]]
    ) -> Any:
        """
        Preprocess inputs for CLIP.

        Args:
            inputs: Images or text strings

        Returns:
            Preprocessed inputs ready for model
        """
        # Handle text inputs
        if isinstance(inputs, str):
            inputs = [inputs]

        if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
            # Text inputs
            if self.use_huggingface:
                return self.processor(text=inputs, return_tensors="pt", padding=True)
            else:
                return clip.tokenize(inputs)

        # Handle image inputs
        if isinstance(inputs, Image.Image):
            inputs = [inputs]

        if self.use_huggingface:
            return self.processor(images=inputs, return_tensors="pt")
        else:
            # Use CLIP preprocessing
            processed = torch.stack([self.preprocess(img) for img in inputs])
            return processed

    async def forward(self, inputs: Any) -> torch.Tensor:
        """
        Run CLIP forward pass.

        Args:
            inputs: Preprocessed inputs

        Returns:
            Embeddings or features
        """
        if not self.clip_model:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Move inputs to device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.config.device)
        elif isinstance(inputs, dict):
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            if self.use_huggingface:
                outputs = self.clip_model(**inputs)
                return outputs
            else:
                # Determine if image or text
                if inputs.dim() == 4:  # Image tensor (B, C, H, W)
                    features = self.clip_model.encode_image(inputs)
                else:  # Text tokens
                    features = self.clip_model.encode_text(inputs)
                return features

    async def postprocess(self, outputs: torch.Tensor) -> np.ndarray:
        """
        Postprocess CLIP outputs.

        Args:
            outputs: Model outputs

        Returns:
            Numpy array of embeddings
        """
        if self.use_huggingface:
            # Extract embeddings from HuggingFace output
            if hasattr(outputs, "image_embeds"):
                embeddings = outputs.image_embeds
            elif hasattr(outputs, "text_embeds"):
                embeddings = outputs.text_embeds
            else:
                embeddings = outputs.last_hidden_state.mean(dim=1)
        else:
            embeddings = outputs

        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # Convert to numpy
        embeddings_np = embeddings.cpu().numpy()

        return embeddings_np

    async def encode_images(
        self, images: Union[Image.Image, List[Image.Image]], return_tensor: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate embeddings for images.

        Args:
            images: Input images
            return_tensor: Return PyTorch tensor instead of numpy

        Returns:
            Image embeddings
        """
        if self.state != ModelState.READY:
            await self.load()

        # Process images
        preprocessed = await self.preprocess(images)
        features = await self.forward(preprocessed)
        embeddings = await self.postprocess(features)

        if return_tensor:
            return torch.from_numpy(embeddings).to(self.config.device)

        return embeddings

    async def encode_text(
        self, texts: Union[str, List[str]], return_tensor: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate embeddings for text.

        Args:
            texts: Input texts
            return_tensor: Return PyTorch tensor instead of numpy

        Returns:
            Text embeddings
        """
        if self.state != ModelState.READY:
            await self.load()

        # Check cache for single text
        if isinstance(texts, str) and texts in self._text_embeddings_cache:
            cached = self._text_embeddings_cache[texts]
            if return_tensor:
                return cached
            return cached.cpu().numpy()

        # Process texts
        preprocessed = await self.preprocess(texts)
        features = await self.forward(preprocessed)
        embeddings = await self.postprocess(features)

        if return_tensor:
            return torch.from_numpy(embeddings).to(self.config.device)

        return embeddings

    async def _encode_text(self, text: str) -> torch.Tensor:
        """Internal method to encode single text to tensor."""
        tokens = clip.tokenize([text]).to(self.config.device)
        with torch.no_grad():
            embedding = self.clip_model.encode_text(tokens)
            if self.normalize_embeddings:
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding[0]  # Return single embedding

    async def compute_similarity(
        self,
        images: Union[Image.Image, List[Image.Image]],
        texts: Union[str, List[str]],
    ) -> np.ndarray:
        """
        Compute similarity between images and texts.

        Args:
            images: Input images
            texts: Input texts

        Returns:
            Similarity matrix (num_images x num_texts)
        """
        # Get embeddings
        image_embeddings = await self.encode_images(images, return_tensor=True)
        text_embeddings = await self.encode_text(texts, return_tensor=True)

        # Ensure 2D tensors
        if image_embeddings.dim() == 1:
            image_embeddings = image_embeddings.unsqueeze(0)
        if text_embeddings.dim() == 1:
            text_embeddings = text_embeddings.unsqueeze(0)

        # Compute cosine similarity
        similarity = torch.matmul(image_embeddings, text_embeddings.T)

        return similarity.cpu().numpy()

    async def score_mug_positioning(
        self, image: Image.Image, custom_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Score mug positioning quality using CLIP.

        Args:
            image: Input image
            custom_prompts: Optional custom prompts to use

        Returns:
            Positioning scores and analysis
        """
        prompts = custom_prompts or self.MUG_PROMPTS

        # Compute similarities
        similarities = await self.compute_similarity(image, prompts)
        similarities = similarities.flatten()  # Single image

        # Create scores dictionary
        scores = {prompt: float(sim) for prompt, sim in zip(prompts, similarities)}

        # Compute aggregate metrics
        analysis = {
            "scores": scores,
            "average_score": float(np.mean(similarities)),
            "max_score": float(np.max(similarities)),
            "min_score": float(np.min(similarities)),
            "best_prompt": prompts[np.argmax(similarities)],
            "worst_prompt": prompts[np.argmin(similarities)],
            "confidence": float(np.std(similarities)),  # Lower std = more consistent
        }

        # Categorize positioning quality
        avg_score = analysis["average_score"]
        if avg_score > 0.3:
            analysis["quality"] = "excellent"
        elif avg_score > 0.25:
            analysis["quality"] = "good"
        elif avg_score > 0.2:
            analysis["quality"] = "fair"
        else:
            analysis["quality"] = "poor"

        return analysis

    async def find_optimal_regions(
        self, image: Image.Image, grid_size: int = 8, region_size: int = 224
    ) -> List[Dict[str, Any]]:
        """
        Find optimal regions for mug placement using sliding window.

        Args:
            image: Input image
            grid_size: Number of regions per dimension
            region_size: Size of each region

        Returns:
            List of optimal regions with scores
        """
        width, height = image.size
        stride_x = (width - region_size) // (grid_size - 1)
        stride_y = (height - region_size) // (grid_size - 1)

        regions = []
        region_images = []
        region_coords = []

        # Extract regions
        for i in range(grid_size):
            for j in range(grid_size):
                x = min(i * stride_x, width - region_size)
                y = min(j * stride_y, height - region_size)

                region = image.crop((x, y, x + region_size, y + region_size))
                region_images.append(region)
                region_coords.append((x, y, x + region_size, y + region_size))

        # Batch encode regions
        region_embeddings = await self.encode_images(region_images, return_tensor=True)

        # Get optimal placement embedding (average of good prompts)
        optimal_prompts = [
            "a clean surface perfect for placing a mug",
            "an empty table area suitable for a cup",
            "a clear space ideal for beverage placement",
        ]
        optimal_embedding = await self.encode_text(optimal_prompts, return_tensor=True)
        optimal_embedding = optimal_embedding.mean(dim=0, keepdim=True)

        # Compute similarities
        similarities = torch.matmul(region_embeddings, optimal_embedding.T).squeeze()

        # Create region results
        for idx, (coords, sim) in enumerate(zip(region_coords, similarities)):
            regions.append(
                {
                    "region_id": idx,
                    "coordinates": {
                        "x1": coords[0],
                        "y1": coords[1],
                        "x2": coords[2],
                        "y2": coords[3],
                    },
                    "center": {
                        "x": (coords[0] + coords[2]) / 2,
                        "y": (coords[1] + coords[3]) / 2,
                    },
                    "score": float(sim),
                    "suitable": float(sim) > 0.25,
                }
            )

        # Sort by score
        regions.sort(key=lambda x: x["score"], reverse=True)

        return regions

    async def generate_positioning_embedding(
        self, image: Image.Image, context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Generate specialized embedding for mug positioning.

        Args:
            image: Input image
            context: Optional context (e.g., detected objects, scene info)

        Returns:
            Positioning-optimized embedding
        """
        # Get base image embedding
        base_embedding = await self.encode_images(image, return_tensor=True)

        # If context provided, create context-aware embedding
        if context:
            context_prompts = []

            # Add context-based prompts
            if "detected_objects" in context:
                objects = context["detected_objects"]
                if "table" in objects or "desk" in objects:
                    context_prompts.append("a mug on a table surface")
                if "keyboard" in objects or "computer" in objects:
                    context_prompts.append("a coffee mug near a workspace")
                if "plate" in objects or "food" in objects:
                    context_prompts.append("a beverage with a meal")

            if "scene_type" in context:
                scene = context["scene_type"]
                if scene == "office":
                    context_prompts.append("a mug in an office setting")
                elif scene == "kitchen":
                    context_prompts.append("a mug in a kitchen environment")
                elif scene == "cafe":
                    context_prompts.append("a cup in a cafe atmosphere")

            # Get context embeddings
            if context_prompts:
                context_embeddings = await self.encode_text(
                    context_prompts, return_tensor=True
                )
                context_embedding = context_embeddings.mean(dim=0)

                # Combine with base embedding (weighted average)
                alpha = 0.7  # Weight for image embedding
                combined_embedding = (
                    alpha * base_embedding + (1 - alpha) * context_embedding
                )
                combined_embedding = combined_embedding / combined_embedding.norm(
                    dim=-1, keepdim=True
                )

                return combined_embedding.cpu().numpy()

        return base_embedding.cpu().numpy()

    async def _create_dummy_input(self) -> Any:
        """Create dummy input for warmup."""
        # Create dummy images
        batch_size = min(4, self.config.max_batch_size)
        dummy_images = []

        for _ in range(batch_size):
            # Random image with CLIP input size
            dummy_img = Image.fromarray(
                np.random.randint(
                    0, 255, (self.input_size, self.input_size, 3), dtype=np.uint8
                )
            )
            dummy_images.append(dummy_img)

        return await self.preprocess(dummy_images)
