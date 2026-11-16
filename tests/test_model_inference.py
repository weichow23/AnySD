"""Exercise inference routing and configuration without loading models or weights.

Only dependency boundaries are stubbed; the complete model.py module is imported
and its real configuration, adapter routing, and checkpoint methods are executed.
These tests do not replace inference tests on CUDA, MPS, or CPU hardware.
"""

from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch, sentinel


MODEL_PATH = Path(__file__).resolve().parents[1] / "anysd" / "src" / "model.py"


@contextmanager
def load_model(*, cuda=False, mps=False, env=None):
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=Mock(return_value=cuda))
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=Mock(return_value=mps)))
    torch.float16 = sentinel.float16
    torch.float32 = sentinel.float32
    torch.device = lambda name: SimpleNamespace(type=name.split(":", 1)[0])
    torch.nn = SimpleNamespace(
        Module=object,
        Parameter=lambda data: SimpleNamespace(data=data),
    )
    torch.randn = Mock(name="randn")
    torch.load = Mock(name="load")
    torch.no_grad = torch.inference_mode = lambda: lambda function: function

    dependencies = {
        "anysd.ip_adapter.ip_adapter": ["Resampler"],
        "diffusers.models.embeddings": ["MultiIPAdapterImageProjection"],
        "anysd.ip_adapter.attention_processor": ["AttnProcessor2_0"],
        "anysd.src.adapter": ["IPAdapterAttnProAnySD"],
        "transformers": ["CLIPTextModel", "CLIPVisionModelWithProjection"],
        "diffusers": ["AutoencoderKL", "StableDiffusionInstructPix2PixPipeline", "UNet2DConditionModel"],
        "anysd.src.pipe": ["AnySDInstructPix2PixPipeline"],
        "anysd.src.unet": ["UNet2DConditionAnySD"],
    }
    modules = {"torch": torch}
    for name, attributes in dependencies.items():
        parts = name.split(".")
        for index in range(1, len(parts) + 1):
            module_name = ".".join(parts[:index])
            if module_name not in modules:
                modules[module_name] = ModuleType(module_name)
                modules[module_name].__path__ = []
        for attribute in attributes:
            setattr(modules[name], attribute, object)

    environment = os.environ.copy()
    environment.pop("ANYSD_DEVICE", None)
    environment.pop("ANYSD_DTYPE", None)
    environment.update(env or {})
    with patch.dict(sys.modules, modules), patch.dict(os.environ, environment, clear=True):
        spec = importlib.util.spec_from_file_location("_anysd_model_under_test", MODEL_PATH)
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
        yield model, torch


class DeviceConfigurationTests(unittest.TestCase):
    def test_device_priority_and_default_dtype(self):
        for cuda, mps, device, dtype in [
            (True, True, "cuda", sentinel.float16),
            (True, False, "cuda", sentinel.float16),
            (False, True, "mps", sentinel.float16),
            (False, False, "cpu", sentinel.float32),
        ]:
            with self.subTest(cuda=cuda, mps=mps), load_model(cuda=cuda, mps=mps) as (model, _):
                self.assertEqual(model.DEVICE, device)
                self.assertIs(model.DTYPE, dtype)

    def test_empty_overrides_use_device_defaults(self):
        for cuda, mps, device, dtype in [
            (True, True, "cuda", sentinel.float16),
            (False, True, "mps", sentinel.float16),
            (False, False, "cpu", sentinel.float32),
        ]:
            with self.subTest(device=device), load_model(
                cuda=cuda, mps=mps, env={"ANYSD_DEVICE": "", "ANYSD_DTYPE": ""}
            ) as (model, _):
                self.assertEqual(model.DEVICE, device)
                self.assertIs(model.DTYPE, dtype)

    def test_explicit_device_and_dtype_overrides(self):
        for device, dtype_override, expected_dtype in [
            ("cpu:0", "", sentinel.float32),
            ("mps", "float32", sentinel.float32),
            ("cuda:1", "float32", sentinel.float32),
        ]:
            with self.subTest(device=device), load_model(
                cuda=True, mps=True,
                env={"ANYSD_DEVICE": device, "ANYSD_DTYPE": dtype_override},
            ) as (model, torch):
                self.assertEqual(model.DEVICE, device)
                self.assertIs(model.DTYPE, expected_dtype)
                torch.cuda.is_available.assert_not_called()
                torch.backends.mps.is_available.assert_not_called()


class AdapterRoutingTests(unittest.TestCase):
    @staticmethod
    def make_pipeline(model):
        pipeline = model.AnySDPipeline.__new__(model.AnySDPipeline)
        pipeline.active_adapter = "general"
        pipeline.adapters_list = {"visual_reference": "/cache/experts/visual_ref.bin"}
        pipeline.sd_pipe = Mock(name="expert_pipeline")
        pipeline.sd_pipe.return_value.images = [sentinel.expert_image]
        pipeline.general_sd_pipe = SimpleNamespace(edit=Mock(return_value=[sentinel.general_image]))
        pipeline.generator = sentinel.generator
        pipeline._process_ref_image_cfg = Mock(return_value=sentinel.reference_embeds)
        return pipeline

    def test_expert_general_expert_routes_and_only_loads_expert_adapters(self):
        with load_model() as (model, _):
            pipeline = self.make_pipeline(model)
            for adapter, expected_image, load_count in [
                ("general", sentinel.general_image, 0),
                ("visual_reference", sentinel.expert_image, 1),
                ("general", sentinel.general_image, 1),
                ("visual_reference", sentinel.expert_image, 2),
                ("visual_reference", sentinel.expert_image, 2),
            ]:
                with self.subTest(adapter=adapter, load_count=load_count):
                    images = pipeline(adapter_name=adapter, prompt="edit", original_image=sentinel.input_image)
                    self.assertEqual(images, [expected_image])
                    self.assertEqual(pipeline.active_adapter, adapter)
                    self.assertEqual(pipeline.sd_pipe.load_ip_adapter.call_count, load_count)
            pipeline.sd_pipe.load_ip_adapter.assert_has_calls([
                call("/cache/", subfolder="experts", weight_name="visual_ref.bin"),
                call("/cache/", subfolder="experts", weight_name="visual_ref.bin"),
            ])
            self.assertEqual(pipeline.general_sd_pipe.edit.call_count, 2)
            self.assertEqual(pipeline.sd_pipe.call_count, 3)
            self.assertEqual(pipeline.list_adapters(), ["visual_reference", "general"])

    def test_unknown_adapter_does_not_change_state_or_run_inference(self):
        with load_model() as (model, _):
            pipeline = self.make_pipeline(model)
            pipeline.active_adapter = "visual_reference"
            with self.assertRaisesRegex(AssertionError, "Adapter 'unknown' not loaded"):
                pipeline(adapter_name="unknown")
            self.assertEqual(pipeline.active_adapter, "visual_reference")
            pipeline.sd_pipe.load_ip_adapter.assert_not_called()
            pipeline.sd_pipe.assert_not_called()
            pipeline.general_sd_pipe.edit.assert_not_called()


class CheckpointLoadingTests(unittest.TestCase):
    def test_task_embeddings_load_on_cpu_then_move_to_pipeline_device_and_dtype(self):
        for device, dtype in [("mps", sentinel.float16), ("cuda", sentinel.float16), ("cpu", sentinel.float32)]:
            with self.subTest(device=device), load_model() as (model, torch):
                pipeline = model.AnySDPipeline.__new__(model.AnySDPipeline)
                pipeline.sd_pipe = SimpleNamespace(device=device, dtype=dtype)
                loaded = Mock(name="checkpoint_tensor")
                loaded.to.return_value = sentinel.converted_embeddings
                torch.load.return_value = loaded

                pipeline._load_task_embs("/cache/experts/task_embs.bin")

                torch.load.assert_called_once_with("/cache/experts/task_embs.bin", map_location="cpu")
                loaded.to.assert_called_once_with(device=device, dtype=dtype)
                self.assertIs(pipeline.sd_pipe.task_embs.data, sentinel.converted_embeddings)


if __name__ == "__main__":
    unittest.main()
