"""Exercise the installed v1 package without downloading model weights."""

import os

os.environ["FACETORCH_OFFLINE"] = "1"

import torch
import torchvision

from facetorch import FaceAnalyzer, load_config


torch_version = tuple(map(int, torch.__version__.split("+")[0].split(".")[:2]))
vision_version = tuple(map(int, torchvision.__version__.split("+")[0].split(".")[:2]))
assert torch_version[0] == 2 and 6 <= torch_version[1] <= 13, torch_version
assert vision_version == (0, torch_version[1] + 15), (torch_version, vision_version)

config = load_config(offline=True)
assert config.analyzer.device == "cpu"
analyzer = FaceAnalyzer(config.analyzer)
result = analyzer.run(
    image_source=torch.zeros((3, 32, 32), dtype=torch.uint8),
    skip_detector=True,
    include_predictors=[],
    include_tensors=True,
)
assert len(result.faces) == 1
assert result.image is not None
assert not analyzer.detector_loaded
assert analyzer.loaded_predictors == ()
