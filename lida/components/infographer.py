
import logging
from typing import Any, List, Union
import PIL
from peacasso.generator import ImageGenerator
from peacasso.datamodel import GeneratorConfig, ModelConfig
from peacasso.utils import base64_to_pil, pil_to_base64
import torch

logger = logging.getLogger(__name__)


class Infographer():
    """Generat infographics given a visualization and a summary of data"""

    def __init__(self, model_config: ModelConfig = None) -> None:
        self.model = None
        self.model_config = model_config or ModelConfig(
            device="cuda:2",
            model="runwayml/stable-diffusion-v1-5",
            revision="main"
        )

    def load_model(self) -> None:
        """Load image generator model from config"""
        self.model = ImageGenerator(model_config=self.model_config)

    def generate(
            self, visualization: Union[torch.FloatTensor, PIL.Image.Image, str],
            n: int, style_prompt: Union[str, List[str]] = "line art pastel",
            return_pil: bool = True
    ) -> List[Any]:
        """Generate a an infographic, given a visualization and style"""

        if isinstance(visualization, str):
            try:
                visualization, _ = base64_to_pil(visualization)
            except Exception as pil_exception:
                logger.error(pil_exception)
                raise ValueError(
                    f'Could not convert provided visualization to PIL image, {str(pil_exception)}') from pil_exception
            self.load_model()

        gen_config = GeneratorConfig(
            prompt=style_prompt,
            num_images=n,
            width=512,
            height=512,
            guidance_scale=7.5,
            num_inference_steps=50,
            init_image=visualization,
            return_intermediates=False,
            seed=2147483647,
            use_prompt_weights=False,
            negative_prompt="text, background shapes or lines, title, words, characters, titles, letters",
            strength=0.6,
            filter_nsfw=False)

        result = self.model.generate(gen_config)
        if not return_pil:
            result["images"] = [pil_to_base64(img) for img in result["images"]]
        return result
