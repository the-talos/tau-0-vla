"""Device-agnostic behaviour of the XPU / CUDA / CPU plumbing.

The XPU path is exercised on shared XPU hardware (the parity test at the bottom),
but everything else runs anywhere: detection order, the adapter dispatch behind
``synchronize``/``empty_cache``, per-device autocast dtype, the deploy warmup RNG
helper, the trainer's save/cleanup paths and the proposal CLI's device default.
Each test pins one behaviour a regression previously broke.

No test here needs a checkpoint: the model-level parity test builds a tiny
random-weight ``Tau0VLAModel`` from config, so it costs seconds on CPU and XPU.
"""

import types
import unittest
from unittest import mock

import torch
from torch import nn

from tau0_vla.utils import device as dev
from tau0_vla.utils.profiling import cuda_sync


def _patch_availability(*, xpu: bool, cuda: bool):
    """Make ``is_available`` report exactly the given backends."""

    def fake_is_available(device_type: str) -> bool:
        if device_type == "xpu":
            return xpu
        if device_type == "cuda":
            return cuda
        return device_type == "cpu"

    return mock.patch.object(dev, "is_available", side_effect=fake_is_available)


class DeviceDetectionTest(unittest.TestCase):
    def test_xpu_preferred_over_cuda(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")
            self.assertEqual(dev.available_device_types(), ["xpu", "cuda", "cpu"])

    def test_cuda_when_no_xpu(self):
        with _patch_availability(xpu=False, cuda=True):
            self.assertEqual(dev.default_device_type(), "cuda")

    def test_cpu_when_no_accelerator(self):
        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.default_device_type(), "cpu")
            self.assertEqual(dev.resolve_device(None), torch.device("cpu"))

    def test_default_device_carries_index(self):
        with _patch_availability(xpu=True, cuda=False):
            self.assertEqual(dev.default_device(1), torch.device("xpu:1"))
            # A CPU has no device index; dropping it keeps torch.device("cpu:0") out of logs.
            with _patch_availability(xpu=False, cuda=False):
                self.assertEqual(dev.default_device(1), torch.device("cpu"))

    def test_explicit_device_wins_over_detection(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda:1"))
            self.assertEqual(dev.resolve_device(torch.device("xpu:1")), torch.device("xpu:1"))

    def test_device_type_from_tensor_string_and_device(self):
        self.assertEqual(dev.device_type("xpu:0"), "xpu")
        self.assertEqual(dev.device_type(torch.device("cuda")), "cuda")
        self.assertEqual(dev.device_type(torch.zeros(1)), "cpu")

    def test_device_count_follows_detected_backend(self):
        with _patch_availability(xpu=True, cuda=False):
            with mock.patch.object(torch.xpu, "device_count", return_value=2) as xpu_count:
                self.assertEqual(dev.device_count(), 2)
                xpu_count.assert_called_once()
        with _patch_availability(xpu=False, cuda=True):
            with mock.patch.object(torch.cuda, "device_count", return_value=8) as cuda_count:
                self.assertEqual(dev.device_count(), 8)
                cuda_count.assert_called_once()
        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.device_count(), 0)


class SynchronizeTest(unittest.TestCase):
    """A CUDA wheel raises 'Torch not compiled with CUDA enabled' on synchronize.

    Every call site therefore has to go through the adapter, which must never touch
    a backend that is not there.
    """

    def test_dispatches_to_xpu(self):
        with _patch_availability(xpu=True, cuda=True):
            with (
                mock.patch.object(torch.xpu, "synchronize") as xpu_sync,
                mock.patch.object(torch.cuda, "synchronize") as cuda_sync_mock,
            ):
                dev.synchronize()
        xpu_sync.assert_called_once()
        cuda_sync_mock.assert_not_called()

    def test_dispatches_to_cuda(self):
        with _patch_availability(xpu=False, cuda=True):
            with mock.patch.object(torch.cuda, "synchronize") as cuda_sync_mock:
                dev.synchronize()
        cuda_sync_mock.assert_called_once()

    def test_noop_without_accelerator(self):
        with _patch_availability(xpu=False, cuda=False):
            with mock.patch.object(torch.cuda, "synchronize") as cuda_sync_mock:
                dev.synchronize()
        cuda_sync_mock.assert_not_called()

    def test_explicit_device_overrides_detection(self):
        with _patch_availability(xpu=True, cuda=True):
            with (
                mock.patch.object(torch.xpu, "synchronize") as xpu_sync,
                mock.patch.object(torch.cuda, "synchronize") as cuda_sync_mock,
            ):
                dev.synchronize("cuda:0")
        cuda_sync_mock.assert_called_once()
        xpu_sync.assert_not_called()

    def test_profiling_alias_stays_device_agnostic(self):
        with _patch_availability(xpu=True, cuda=False):
            with mock.patch.object(torch.xpu, "synchronize") as xpu_sync:
                cuda_sync()
        xpu_sync.assert_called_once()


class EmptyCacheTest(unittest.TestCase):
    def test_dispatches_to_xpu_then_cuda_then_nothing(self):
        with _patch_availability(xpu=True, cuda=True):
            with (
                mock.patch.object(torch.xpu, "empty_cache") as xpu_cache,
                mock.patch.object(torch.cuda, "empty_cache") as cuda_cache,
            ):
                dev.empty_cache()
        xpu_cache.assert_called_once()
        cuda_cache.assert_not_called()

        with _patch_availability(xpu=False, cuda=True):
            with mock.patch.object(torch.cuda, "empty_cache") as cuda_cache:
                dev.empty_cache()
        cuda_cache.assert_called_once()

        with _patch_availability(xpu=False, cuda=False):
            with mock.patch.object(torch.cuda, "empty_cache") as cuda_cache:
                dev.empty_cache()
        cuda_cache.assert_not_called()


class AutocastDtypeTest(unittest.TestCase):
    """Autocast state is per device type; the CUDA getter lies on XPU."""

    def test_cuda_getter_is_not_used_for_xpu(self):
        with _patch_availability(xpu=True, cuda=True):
            with (
                mock.patch("torch.get_autocast_gpu_dtype", return_value=torch.float16) as cuda_dtype,
                mock.patch("torch.get_autocast_dtype", return_value=torch.bfloat16) as any_dtype,
            ):
                self.assertEqual(dev.autocast_dtype("xpu"), torch.bfloat16)
        any_dtype.assert_called_once_with("xpu")
        cuda_dtype.assert_not_called()

    def test_reports_cuda_dtype_for_cuda(self):
        with _patch_availability(xpu=False, cuda=True):
            with mock.patch("torch.get_autocast_dtype", return_value=torch.float16) as any_dtype:
                self.assertEqual(dev.autocast_dtype("cuda"), torch.float16)
        any_dtype.assert_called_once_with("cuda")

    def test_compute_dtype_follows_active_autocast_device(self):
        """The FA2 shim casts fp32 LayerNorm output with the *active device* dtype."""
        linear = nn.Linear(4, 4, dtype=torch.bfloat16)
        query = torch.randn(1, 2, 4, 8)

        with _patch_availability(xpu=True, cuda=True):
            with (
                mock.patch.object(dev, "is_autocast_enabled", return_value=True),
                mock.patch.object(dev, "autocast_dtype", return_value=torch.bfloat16),
            ):
                self.assertEqual(dev.resolve_compute_dtype(linear, query), torch.bfloat16)

    def test_compute_dtype_falls_back_to_weight_dtype_without_autocast(self):
        linear = nn.Linear(4, 4, dtype=torch.bfloat16)
        query = torch.randn(1, 2, 4, 8)

        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.resolve_compute_dtype(linear, query), torch.bfloat16)

    def test_compute_dtype_falls_back_to_input_dtype_without_linear(self):
        module = nn.ReLU()
        query = torch.randn(1, 2, 4, 8)
        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.resolve_compute_dtype(module, query), torch.float32)


class WarmupRngTest(unittest.TestCase):
    """Deploy warmup samples device noise, so it saves and restores device RNG state."""

    def test_none_without_accelerator(self):
        with _patch_availability(xpu=False, cuda=False):
            self.assertIsNone(dev.get_device_rng_state())

    def test_round_trip_on_detected_backend(self):
        with _patch_availability(xpu=True, cuda=False):
            with (
                mock.patch.object(torch.xpu, "get_rng_state_all", return_value=["state"]) as get_state,
                mock.patch.object(torch.xpu, "set_rng_state_all") as set_state,
            ):
                state = dev.get_device_rng_state()
                dev.set_device_rng_state(state)
        self.assertEqual(state, ["state"])
        get_state.assert_called_once()
        set_state.assert_called_once_with(["state"])

    def test_set_is_noop_for_none(self):
        with _patch_availability(xpu=True, cuda=False):
            with mock.patch.object(torch.xpu, "set_rng_state_all") as set_state:
                dev.set_device_rng_state(None)
        set_state.assert_not_called()


class TrainerDevicePathsTest(unittest.TestCase):
    """The trainer must not call CUDA directly: both entry points crashed on XPU."""

    def test_save_helper_syncs_detected_backend(self):
        from tau0_vla.trainer.trainer import safe_save_model_for_hf_trainer

        trainer = mock.Mock()
        trainer.deepspeed = object()

        with _patch_availability(xpu=True, cuda=False):
            with (
                mock.patch.object(torch.xpu, "synchronize") as xpu_sync,
                mock.patch.object(torch.cuda, "synchronize") as cuda_sync_mock,
            ):
                safe_save_model_for_hf_trainer(trainer, "/tmp/out")
        xpu_sync.assert_called_once()
        cuda_sync_mock.assert_not_called()
        trainer.save_model.assert_called_once_with("/tmp/out")

    def test_manual_cleanup_uses_detected_backend(self):
        from tau0_vla.trainer.trainer import Tau0VLATrainer

        stub = types.SimpleNamespace(
            args=types.SimpleNamespace(torch_empty_cache_steps=10),
            state=types.SimpleNamespace(global_step=10),
            _manual_memory_cleanup_last_step=None,
        )

        with _patch_availability(xpu=True, cuda=False):
            with (
                mock.patch("tau0_vla.trainer.trainer.empty_cache") as empty_cache,
                mock.patch.object(torch.cuda, "empty_cache") as cuda_cache,
            ):
                Tau0VLATrainer._maybe_run_manual_memory_cleanup(stub)
        empty_cache.assert_called_once()
        cuda_cache.assert_not_called()
        self.assertEqual(stub._manual_memory_cleanup_last_step, 10)


class ProposalDeviceFlagTest(unittest.TestCase):
    """The high-level CLIs defaulted to 'cuda:0', which an XPU node cannot resolve."""

    def test_device_defaults_to_autodetect(self):
        from tau0_vla.high_level.proposal import evaluate, finetune, infer, serve

        for module in (evaluate, finetune, infer, serve):
            parser = module.build_parser()
            action = next(a for a in parser._actions if a.dest == "device")
            self.assertIsNone(action.default, f"{module.__name__} still hard-codes a device")

    def test_runtime_defaults_to_autodetect(self):
        import inspect

        from tau0_vla.high_level.proposal import runtime

        load_model_sig = inspect.signature(runtime.load_model)
        proposal_sig = inspect.signature(runtime.Proposal.__init__)
        self.assertIsNone(load_model_sig.parameters["device"].default)
        self.assertIsNone(proposal_sig.parameters["device"].default)


class DeployPolicyDeviceTest(unittest.TestCase):
    def test_from_checkpoint_resolves_detected_device(self):
        import inspect

        from deploy.policy import Tau0VLAPolicy

        # The resolution itself is covered above; pin that the policy routes through it
        # rather than repeating a cuda/cpu ternary that hides XPU.
        source = inspect.getsource(Tau0VLAPolicy.from_checkpoint)
        self.assertIn("resolve_device", source)
        self.assertNotIn('torch.device("cuda" if', source)


@unittest.skipIf(not dev.is_available("xpu"), "XPU not available")
class XpuParityTest(unittest.TestCase):
    """CPU vs XPU parity for a tiny random-weight Tau0VLA model (no checkpoint)."""

    @staticmethod
    def _tiny_model_and_batch(dtype):
        from transformers.models.qwen3_5.configuration_qwen3_5 import (
            Qwen3_5Config,
            Qwen3_5TextConfig,
            Qwen3_5VisionConfig,
        )

        from tau0_vla.models.vision_language_action_models.tau0_vla import Tau0VLAConfig, Tau0VLAModel

        text = Qwen3_5TextConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            layer_types=["full_attention", "full_attention"],
            max_position_embeddings=512,
        )
        vision = Qwen3_5VisionConfig(
            depth=1,
            hidden_size=32,
            out_hidden_size=64,
            num_heads=4,
            in_chans=3,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )
        config = Tau0VLAConfig(
            qwenvl_config=Qwen3_5Config(vision_config=vision.to_dict(), text_config=text.to_dict()).to_dict(),
            attention_implementation="eager",
            max_action_dim=8,
            max_state_dim=8,
            action_dim=8,
            n_action_steps=4,
            proj_width=64,
            num_steps=2,
        )
        torch.manual_seed(1234)
        model = Tau0VLAModel(config).to(dtype=dtype).eval()

        generator = torch.Generator().manual_seed(0)
        input_ids = torch.randint(3, 256, (1, 12), generator=generator)
        state = torch.randn(1, 1, 8, generator=torch.Generator().manual_seed(1))
        noise = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(7))
        return model, {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "state": state,
            "noise": noise,
        }

    def _sample(self, model, batch, device, dtype):
        moved = {
            k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device) for k, v in batch.items()
        }
        noise = moved.pop("noise")
        with torch.no_grad():
            actions = model.flow_matching.sample_actions(
                state=moved["state"].squeeze(1),
                input_ids=moved["input_ids"],
                attention_mask=moved["attention_mask"],
                noise=noise,
            )[:, :, : model.config.action_dim]
        torch.xpu.synchronize()
        return actions.detach().float().cpu()

    def test_float32_actions_match_cpu(self):
        model, batch = self._tiny_model_and_batch(torch.float32)
        cpu_actions = self._sample(model, batch, torch.device("cpu"), torch.float32)
        xpu_model = model.to("xpu")
        xpu_actions = self._sample(xpu_model, batch, torch.device("xpu:0"), torch.float32)

        self.assertEqual(xpu_actions.shape, cpu_actions.shape)
        mse = torch.mean((cpu_actions - xpu_actions) ** 2).item()
        self.assertLess(mse, 1e-6, f"CPU vs XPU float32 MSE {mse:.3e} exceeds tolerance")

    def test_bfloat16_actions_match_cpu(self):
        model, batch = self._tiny_model_and_batch(torch.bfloat16)
        cpu_actions = self._sample(model, batch, torch.device("cpu"), torch.bfloat16)
        xpu_model = model.to("xpu")
        xpu_actions = self._sample(xpu_model, batch, torch.device("xpu:0"), torch.bfloat16)

        mse = torch.mean((cpu_actions - xpu_actions) ** 2).item()
        self.assertLess(mse, 1e-3, f"CPU vs XPU bfloat16 MSE {mse:.3e} exceeds tolerance")


if __name__ == "__main__":
    unittest.main()
