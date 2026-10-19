"""The training loop: a HuggingFace ``Trainer`` subclass and its callbacks.

- :class:`Tau0VLATrainer` — drives the single action route, owns the data
  iterator, and handles resume.
- :class:`DummyDataLoader` — satisfies the iteration interface ``Trainer``
  expects; real batches come from the iterator the trainer holds.
- :class:`FinchDataSpecCallback` — mirrors ``finch_data_spec/`` and
  ``policy_manifest.json`` into every ``checkpoint-N/``, so a checkpoint
  carries the data contract inference needs.
- :class:`SaveProcessorCallback` — writes the Processor alongside each
  checkpoint.
- :func:`safe_save_model_for_hf_trainer` — DeepSpeed-aware model save.
"""

import csv
import gc
import logging
import os
import time
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
from accelerate.utils import DistributedType
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint

from tau0_vla.utils.device import empty_cache, is_available, synchronize
from tau0_vla.utils.utils import rank0_print

logger = logging.getLogger(__name__)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Metric prefix for the single action route. The historical route name "vla" is
# kept deliberately so that ``vla_loss`` / ``vla_epoch`` / ``vla_*_ms`` stay
# comparable across versions on existing wandb dashboards.
_ROUTE_NAME = "vla"

_SYNCED_LOSS_LOG_KEYS = ("vla_loss",)

_PROFILE_TIME_KEYS = (
    "data_time_ms",
    "to_device_time_ms",
    "forward_time_ms",
    "backward_time_ms",
    "optimizer_time_ms",
    "lr_scheduler_time_ms",
    "step_time_ms",
)


class Tau0VLATrainer(Trainer):
    """Custom ``Trainer`` driving the single action route.

    Extends ``transformers.Trainer`` with:

    - A training ``DataLoader`` injected directly by ``train.py``, which lets a
      ``physically_split`` dataset skip ``accelerator.prepare()`` and so avoids
      sharding twice — once by the dataset, once by accelerate.
    - A custom forward function (``forward_fn``); this class owns loss scaling
      and the backward pass.
    - Sub-loss tracking and reporting (``vla_loss``).
    - The route's own epoch counter, reported through logs and wandb
      (``vla_epoch``).
    - Persisted epoch / steps_in_epoch, replayed on resume with
      ``skip_first_batches``.
    - Differential multimodal learning rates (separate LRs for the vision
      encoder and the VL projector).
    - Optional per-step profiling, writing log means plus a per-rank CSV.

    Args:
        *args: Positional arguments forwarded to ``transformers.Trainer``.
        train_dataloader: The training ``DataLoader``. Built by ``train.py``; this
            class no longer constructs a sampler or DataLoader itself.
        forward_fn: Custom forward function with the signature
            ``(model, inputs, loss_scale=1.0) -> (loss, outputs_dict)``.
        log_metrics_sync_across_ranks: Whether ``log()`` all-reduces its loss and
            epoch metrics into a global average across ranks.
        **kwargs: Keyword arguments forwarded to ``transformers.Trainer``.
    """

    def __init__(
        self,
        *args,
        train_dataloader: Optional[DataLoader] = None,
        forward_fn: Optional[callable] = None,
        log_metrics_sync_across_ranks: bool = True,
        **kwargs,
    ):
        """Initialize the single-route Trainer.

        Args:
            *args: Positional arguments forwarded to the base class.
            train_dataloader: The training ``DataLoader``, built and passed in by
                ``train.py``.
            forward_fn: Custom forward function taking
                ``(model, inputs, loss_scale)`` and returning either
                ``(total_loss, outputs_dict)`` or just ``loss``.
            log_metrics_sync_across_ranks: Whether to aggregate log metrics across
                ranks.
            **kwargs: Remaining arguments forwarded to the HF Trainer.
        """
        self.train_dataloader = train_dataloader
        self.forward_fn = forward_fn
        # Whether log() all-reduces its loss / epoch metrics into a global mean.
        # When False it uses this rank's local value — rank 0's, since only rank 0
        # writes logs — skipping both the all-reduce and the device-to-host sync.
        # Worth it for short steps at large world sizes.
        self._log_metrics_sync_across_ranks = log_metrics_sync_across_ranks
        self._enable_step_profiling = False
        self._step_profiling_cuda_sync = False
        self._profile_stats: dict[str, dict[str, float]] = {
            key: {"total": 0.0, "count": 0} for key in _PROFILE_TIME_KEYS
        }
        # Count metrics are registered on demand (the collator reports them under
        # profile_* keys), so there is nothing to preallocate.
        self._profile_count_stats: dict[str, float] = {}
        self._profile_wrapped_dataloader = None
        self._optimizer_step_profile_wrapped = False
        self._lr_scheduler_step_profile_wrapped = False
        self._profile_csv_enabled = False
        self._profile_step_row: dict[str, float] = {}
        self._profile_csv_rows: list[dict[str, float]] = []
        self._profile_csv_file = None
        self._profile_csv_writer = None
        self._profile_csv_micro_step = 0

        # The route's own data-scheduling state: iterator plus epoch counter,
        # maintained here and written into the checkpoint.
        self._train_iterator = None
        self._train_iterator_initialized = False
        self._data_epoch = 0
        self._steps_in_epoch = 0
        self._resume_checkpoint_path: Optional[str] = None
        self._data_state_restored: bool = False

        self.loss_tracker: dict[str, dict[str, Any]] = {}

        # Force ignore_data_skip=True: resume replays data through this class's own
        # _load_data_state / skip_first_batches, not HF's default path — which
        # would re-``iter()`` the DataLoader we injected.
        training_args = kwargs.get("args") or (args[1] if len(args) > 1 else None)
        if training_args is not None and not training_args.ignore_data_skip:
            training_args.ignore_data_skip = True

        super().__init__(*args, **kwargs)
        self._manual_memory_cleanup_last_step: int = -1
        self._python_gc_disabled_by_trainer: bool = False
        self._python_gc_was_enabled: Optional[bool] = None
        self._enable_step_profiling = bool(getattr(self.args, "enable_step_profiling", False))
        self._step_profiling_cuda_sync = bool(getattr(self.args, "step_profiling_cuda_sync", False))
        self._profile_csv_enabled = self._enable_step_profiling and bool(
            getattr(self.args, "step_profiling_all_ranks_csv", False)
        )
        self._base_seed = int(getattr(self.args, "seed", 42))
        # forward_fn computes the complete loss itself and never consumes
        # num_items_in_batch, so HF Trainer's contract requires switching this flag
        # off explicitly. Left on, HF counts tokens in get_batch_samples via
        # labels.ne(-100), passes num_items_in_batch down, and
        # _scale_loss_for_backward then skips its /gradient_accumulation_steps —
        # inflating the effective learning rate by a factor of gas, silently.
        # (Historically the batch was a list, which happened not to trigger the
        # token count at all.)
        self.model_accepts_loss_kwargs = False

    def _wrap_model(self, model, training=True, dataloader=None):
        """Override to manage DDP static_graph mode.

        static_graph=True instead of find_unused_parameters=True:
        - It tells DDP the computation graph is identical every step (parameters
          that go unused stay unused), so no Python autograd hook is needed to
          detect them and backward is not blocked on the Python main thread.

        With a single action route every step runs the same forward path over the
        same parameter set, so the graph really is static and static_graph can be
        enabled unconditionally.
        """
        wrapped = super()._wrap_model(model, training, dataloader)
        if training and isinstance(wrapped, torch.nn.parallel.DistributedDataParallel):
            # Debug: dump DDP parameter indices → names so unused-param errors are diagnosable.
            # DDP indexes by reversed order of model.parameters() that require grad.
            if int(os.environ.get("RANK", "0")) == 0:
                trainable = [(n, p) for n, p in wrapped.module.named_parameters() if p.requires_grad]
                print(f"[DDP debug] total trainable params: {len(trainable)}", flush=True)
                # Print a window around the indices reported by the recent error (608–616)
                # plus first/last few so user can sanity-check ordering.
                for i, (n, p) in enumerate(trainable):
                    if i < 5 or i >= len(trainable) - 5 or 600 <= i <= 620:
                        print(f"[DDP debug] param[{i}] {n} shape={tuple(p.shape)}", flush=True)
            wrapped._set_static_graph()
            logger.info("DDP static_graph=True enabled (single action route, graph is static)")
        return wrapped

    def _log_dataloader_summary(self) -> None:
        """Print a summary of how the training data is sharded and batched."""
        dataset = self.train_dataset
        dl = self.train_dataloader
        if dataset is None or dl is None:
            return

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        n_samples = len(dataset)
        # ``dl.batch_size`` is forced to None by torch's DataLoader contract once
        # ``batch_sampler`` is set, which happens after ``accelerator.prepare``
        # wraps the loader with a DistributedSampler. The real batch size is
        # preserved on the BatchSampler — fall back to that before "unknown".
        batch_size = (
            getattr(dl, "batch_size", None)
            or getattr(getattr(dl, "batch_sampler", None), "batch_size", None)
            or "unknown"
        )
        n_batches = len(dl)
        physically_split = bool(getattr(dataset, "physically_split", False))
        manual_sampler = bool(getattr(dataset, "use_manual_distributed_sampler", False))
        if manual_sampler or (physically_split and world_size > 1):
            # The dataset already sharded itself by rank (physically_split or a
            # manual sampler), so len() is the per-rank size and the global
            # figures can only be estimated.
            rank_samples = str(n_samples)
            global_samples = f"~{n_samples * world_size} (estimated)"
            global_batches = f"~{n_batches * world_size} (estimated)"
        elif world_size > 1:
            # DistributedSampler: the dataset is global, the dataloader per-rank.
            rank_samples = str(n_samples // world_size)
            global_samples = str(n_samples)
            global_batches = str(n_batches * world_size)
        else:
            rank_samples = str(n_samples)
            global_samples = str(n_samples)
            global_batches = str(n_batches)

        rank0_print("=" * 60)
        rank0_print(f"Train data summary (route={_ROUTE_NAME}, rank={rank}, world_size={world_size})")
        rank0_print(
            f"  [{_ROUTE_NAME}] rank_samples={rank_samples}, global_samples={global_samples}, "
            f"batch_size={batch_size}, rank_batches={n_batches}, global_batches={global_batches}, "
            f"physically_split={physically_split}, manual_sampler={manual_sampler}, "
            f"physical_rank={getattr(dataset, 'physical_shard_rank', 'n/a')}/"
            f"{getattr(dataset, 'physical_shard_world_size', 'n/a')}, "
            f"logical_rank={getattr(dataset, 'logical_shard_rank', 'n/a')}/"
            f"{getattr(dataset, 'logical_shard_world_size', 'n/a')}"
        )
        rank0_print("=" * 60)

    def _set_dataloader_epoch(self, epoch: int) -> None:
        """Set the current epoch on the training DataLoader, whatever its type."""
        dataloader = self.train_dataloader
        # Check batch_sampler first
        batch_sampler = getattr(dataloader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
            return
        # The DataLoaderShard accelerator.prepare() returns has set_epoch directly.
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch)
            return
        # A plain DataLoader: its sampler may be a DistributedSampler.
        sampler = getattr(dataloader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
            return
        # A DistributedSampler is sometimes wrapped inside a BatchSampler.
        if batch_sampler is not None:
            inner_sampler = getattr(batch_sampler, "sampler", None)
            if isinstance(inner_sampler, DistributedSampler):
                inner_sampler.set_epoch(epoch)
                return
        # physically_split mode: RandomSampler has no set_epoch, so seed its
        # generator by hand to keep each epoch's permutation reproducible.
        if isinstance(sampler, RandomSampler):
            if sampler.generator is None:
                sampler.generator = torch.Generator()
            sampler.generator.manual_seed(self._base_seed + epoch)
            return

    def _propagate_stride_epoch(self, epoch: int) -> None:
        """Propagate the data epoch to any _FrameStrideDataset in the wrapper
        chain so deterministic stride cycling (do_random_shift=False) uses the
        dataset's own epoch counter, not a global one."""
        if self.train_dataset is not None:
            self._walk_stride_epoch(self.train_dataset, epoch)

    @staticmethod
    def _walk_stride_epoch(obj: Any, epoch: int) -> None:
        if hasattr(obj, "steps_by_frame") and hasattr(obj, "set_epoch"):
            obj.set_epoch(epoch)
        for attr in ("_underlying", "_dataset", "_inner"):
            child = getattr(obj, attr, None)
            if child is not None and child is not obj:
                Tau0VLATrainer._walk_stride_epoch(child, epoch)
        for child in getattr(obj, "datasets", []):
            if hasattr(child, "__getitem__"):
                Tau0VLATrainer._walk_stride_epoch(child, epoch)

    def _init_train_iterator(self) -> None:
        """Build the training iterator and sync the epoch to sampler and stride datasets."""
        self._set_dataloader_epoch(self._data_epoch)
        self._propagate_stride_epoch(self._data_epoch)
        self._train_iterator = iter(self.train_dataloader)
        self._train_iterator_initialized = True
        logger.info(f"Initialized train iterator with length {len(self.train_dataloader)}")

    def get_next_batch(self) -> Dict[str, Any]:
        """Return the next training batch, rolling the epoch over when exhausted.

        The iterator **must** be built lazily — ``iter()`` on the first batch, not
        eagerly in ``__init__``. ``iter(dl)`` immediately spawns ``num_workers``
        persistent workers and starts prefetching, and the resume path then
        rebuilds the DataLoader through ``skip_first_batches``, spawning a second
        full set. The original DataLoader still references the first set, so
        those workers never exit and their prefetch queues stay full: per-node
        worker memory simply doubles. On a 96-node resume every node was
        SIGKILLed by the host OOM killer within seconds of
        "[Resume] skipped N batches" — a real incident, not a hypothetical.
        """
        if not self._train_iterator_initialized:
            self._init_train_iterator()

        try:
            batch = next(self._train_iterator)
            self._steps_in_epoch += 1
            return batch
        except StopIteration:
            return self._reset_and_next()

    def _reset_and_next(self) -> Any:
        """Reset the iterator and advance the epoch counter."""
        self._train_iterator = None
        self._data_epoch += 1
        self._steps_in_epoch = 0

        self._set_dataloader_epoch(self._data_epoch)
        self._propagate_stride_epoch(self._data_epoch)
        logger.info(f"Train dataset finished an epoch. Starting epoch {self._data_epoch}.")
        self._train_iterator = iter(self.train_dataloader)
        batch = next(self._train_iterator)
        self._steps_in_epoch += 1
        return batch

    def _data_state_dict(self) -> Dict[str, Any]:
        """Return serializable data-scheduling state for checkpoint resume."""
        return {
            "base_seed": self._base_seed,
            "rank": dist.get_rank() if dist.is_initialized() else 0,
            "route": _ROUTE_NAME,
            "epoch": int(self._data_epoch),
            "steps_in_epoch": int(self._steps_in_epoch),
        }

    def _load_data_state(self, state: Dict[str, Any]) -> None:
        """Restore epoch counters and fast-forward the iterator to the saved offset."""
        from accelerate import skip_first_batches

        self._data_epoch = int(state.get("epoch", 0))
        self._steps_in_epoch = int(state.get("steps_in_epoch", 0))

        # If the iterator was already built eagerly (the old behaviour, before the
        # deferred path), shut down the original DataLoader's persistent workers
        # first. Otherwise the DataLoader that skip_first_batches rebuilds spawns a
        # second set while the first leaks along with its prefetch queues — enough
        # to exhaust host memory on a multi-node resume, which in practice got
        # every node killed by the OOM killer.
        self._train_iterator = None
        stale_iter = getattr(self.train_dataloader, "_iterator", None)
        if stale_iter is not None:
            try:
                stale_iter._shutdown_workers()
            except Exception:
                pass
            self.train_dataloader._iterator = None

        self._set_dataloader_epoch(self._data_epoch)
        self._propagate_stride_epoch(self._data_epoch)
        if self._steps_in_epoch > 0:
            skipped_dl = skip_first_batches(self.train_dataloader, self._steps_in_epoch)
            self._train_iterator = iter(skipped_dl)
            logger.info(f"[Resume] skipped {self._steps_in_epoch} batches (epoch={self._data_epoch})")
        else:
            self._train_iterator = iter(self.train_dataloader)
        self._train_iterator_initialized = True

    def _get_data_state_path(self, checkpoint_dir: str) -> str:
        rank = dist.get_rank() if dist.is_initialized() else 0
        return os.path.join(checkpoint_dir, f"data_state_rank{rank}.pt")

    def _restore_data_state_if_needed(self) -> None:
        if self._data_state_restored or not self._resume_checkpoint_path:
            return

        state_path = self._get_data_state_path(self._resume_checkpoint_path)
        if not os.path.exists(state_path):
            rank0_print(f"[Resume] Data state not found at {state_path}, starting data scheduling from scratch.")
            self._data_state_restored = True
            return

        state = torch.load(state_path, map_location="cpu")
        self._load_data_state(state)
        self._data_state_restored = True
        rank0_print(f"[Resume] Restored data state from {state_path}")

    def _is_step_profiling_enabled(self) -> bool:
        return bool(getattr(self, "_enable_step_profiling", False))

    def _record_profile_time(self, key: str, elapsed_ms: float) -> None:
        if not self._is_step_profiling_enabled():
            return
        stats = self._profile_stats.setdefault(key, {"total": 0.0, "count": 0})
        stats["total"] += float(elapsed_ms)
        stats["count"] += 1
        if self._profile_csv_enabled:
            # A step may record several times (one per gradient-accumulation
            # micro-step), so timing metrics accumulate into the step's total.
            self._profile_step_row[key] = self._profile_step_row.get(key, 0.0) + float(elapsed_ms)

    def _record_profile_average(self, key: str, value: float) -> None:
        if not self._is_step_profiling_enabled():
            return
        stats = self._profile_stats.setdefault(key, {"total": 0.0, "count": 0})
        stats["total"] += float(value)
        stats["count"] += 1
        if self._profile_csv_enabled:
            self._profile_step_row[key] = float(value)

    def _record_model_profile_outputs(self, outputs: Union[Dict, Any]) -> None:
        if not self._is_step_profiling_enabled() or outputs is None:
            return
        items = outputs.items() if isinstance(outputs, dict) else vars(outputs).items()
        for key, value in items:
            if not key.startswith("profile_") or value is None:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().float().item()
            self._record_profile_average(f"{_ROUTE_NAME}_{key.removeprefix('profile_')}", float(value))

    def _maybe_profile_cuda_sync(self) -> None:
        if self._is_step_profiling_enabled() and getattr(self, "_step_profiling_cuda_sync", False):
            synchronize()

    def _wrap_optimizer_step_for_profiling(self) -> None:
        if (
            not self._is_step_profiling_enabled()
            or getattr(self, "_optimizer_step_profile_wrapped", False)
            or self.optimizer is None
        ):
            return
        step = getattr(self.optimizer, "step", None)
        if step is None or getattr(step, "_tau0_profile_wrapped", False):
            self._optimizer_step_profile_wrapped = True
            return

        def profiled_step(*args, **kwargs):
            self._maybe_profile_cuda_sync()
            t0 = time.perf_counter()
            result = step(*args, **kwargs)
            self._maybe_profile_cuda_sync()
            self._record_profile_time("optimizer_time_ms", (time.perf_counter() - t0) * 1000.0)
            return result

        profiled_step._tau0_profile_wrapped = True
        self.optimizer.step = profiled_step
        self._optimizer_step_profile_wrapped = True

    def _wrap_lr_scheduler_step_for_profiling(self) -> None:
        if (
            not self._is_step_profiling_enabled()
            or getattr(self, "_lr_scheduler_step_profile_wrapped", False)
            or self.lr_scheduler is None
        ):
            return
        step = getattr(self.lr_scheduler, "step", None)
        if step is None or getattr(step, "_tau0_profile_wrapped", False):
            self._lr_scheduler_step_profile_wrapped = True
            return

        def profiled_step(*args, **kwargs):
            self._maybe_profile_cuda_sync()
            t0 = time.perf_counter()
            result = step(*args, **kwargs)
            self._maybe_profile_cuda_sync()
            self._record_profile_time("lr_scheduler_time_ms", (time.perf_counter() - t0) * 1000.0)
            return result

        profiled_step._tau0_profile_wrapped = True
        self.lr_scheduler.step = profiled_step
        self._lr_scheduler_step_profile_wrapped = True

    def _ensure_step_profiling_hooks(self) -> None:
        if not self._is_step_profiling_enabled():
            return
        self._wrap_optimizer_step_for_profiling()
        self._wrap_lr_scheduler_step_for_profiling()

    def _unwrap_step_profiling_hooks(self) -> None:
        """Remove the profiling monkey-patches from optimizer/lr_scheduler.

        The wrappers are local closures stored as instance attributes; the LR
        scheduler's ``state_dict()`` serializes its ``__dict__``, so a wrapped
        ``step`` makes ``torch.save`` fail with "Can't pickle local object".
        Deleting the instance attribute restores the class method; the hooks
        are re-installed by ``_ensure_step_profiling_hooks`` on the next step.
        """
        for obj, flag in (
            (self.optimizer, "_optimizer_step_profile_wrapped"),
            (self.lr_scheduler, "_lr_scheduler_step_profile_wrapped"),
        ):
            if obj is not None and getattr(obj.__dict__.get("step"), "_tau0_profile_wrapped", False):
                del obj.__dict__["step"]
            setattr(self, flag, False)

    _PROFILE_CSV_FLUSH_EVERY = 20

    def _finish_profile_csv_step(self) -> None:
        """Queue the current step's per-rank profile row and periodically write it out.

        Called at the end of every ``training_step``. The pending row holds
        everything recorded since the previous call: the dataloader wait for
        this step's batch (recorded by ``ProfilingDataLoaderProxy`` before
        ``training_step`` runs) plus to_device/forward/backward/step times.
        optimizer/lr_scheduler run after ``training_step`` and therefore land
        on the NEXT row (both are ~0.05 ms, only the attribution shifts).
        """
        if not self._profile_csv_enabled:
            return
        self._profile_csv_micro_step += 1
        row: dict[str, Any] = {
            "global_step": int(self.state.global_step) if self.state is not None else -1,
            "micro_step": self._profile_csv_micro_step,
            "wall_time": round(time.time(), 3),
        }
        row.update({k: round(v, 3) for k, v in self._profile_step_row.items()})
        self._profile_step_row = {}
        self._profile_csv_rows.append(row)
        if len(self._profile_csv_rows) >= self._PROFILE_CSV_FLUSH_EVERY:
            self._flush_profile_csv_rows()

    def _flush_profile_csv_rows(self) -> None:
        if not self._profile_csv_rows:
            return
        if self._profile_csv_writer is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            csv_dir = os.path.join(self.args.output_dir, "step_profile")
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, f"step_profile_rank{rank:05d}.csv")
            # Header = union of keys in the first buffered window. The single
            # action route contributes the same key set every step, so the first
            # _PROFILE_CSV_FLUSH_EVERY rows already cover it; a key that only
            # shows up later is dropped from the CSV.
            fieldnames = ["global_step", "micro_step", "wall_time"]
            extra = sorted({k for r in self._profile_csv_rows for k in r} - set(fieldnames))
            self._profile_csv_file = open(csv_path, "a", newline="")
            self._profile_csv_writer = csv.DictWriter(
                self._profile_csv_file, fieldnames=fieldnames + extra, restval="", extrasaction="ignore"
            )
            if self._profile_csv_file.tell() == 0:
                self._profile_csv_writer.writeheader()
            logger.info("Per-rank step profiling CSV: %s", csv_path)
        self._profile_csv_writer.writerows(self._profile_csv_rows)
        self._profile_csv_rows.clear()
        self._profile_csv_file.flush()

    def _close_profile_csv(self) -> None:
        if not self._profile_csv_enabled:
            return
        try:
            self._flush_profile_csv_rows()
        finally:
            if self._profile_csv_file is not None:
                self._profile_csv_file.close()
                self._profile_csv_file = None
                self._profile_csv_writer = None

    def _maybe_wrap_dataloader_for_profiling(self, dataloader: Any) -> Any:
        if not self._is_step_profiling_enabled():
            return dataloader
        if isinstance(dataloader, ProfilingDataLoaderProxy):
            return dataloader
        profile_wrapped_dataloader = getattr(self, "_profile_wrapped_dataloader", None)
        if (
            isinstance(profile_wrapped_dataloader, ProfilingDataLoaderProxy)
            and profile_wrapped_dataloader._dataloader is dataloader
        ):
            return profile_wrapped_dataloader
        wrapped = ProfilingDataLoaderProxy(dataloader, self)
        self._profile_wrapped_dataloader = wrapped
        logger.info(
            "Step profiling enabled: logging averaged %s over each logging window%s",
            ", ".join(_PROFILE_TIME_KEYS),
            " with CUDA synchronization" if getattr(self, "_step_profiling_cuda_sync", False) else "",
        )
        return wrapped

    def train(self, resume_from_checkpoint: str | bool | None = None, trial=None, ignore_keys_for_eval=None):
        if resume_from_checkpoint is True:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
        self._resume_checkpoint_path = resume_from_checkpoint if isinstance(resume_from_checkpoint, str) else None
        self._data_state_restored = False
        try:
            return super().train(
                resume_from_checkpoint=resume_from_checkpoint, trial=trial, ignore_keys_for_eval=ignore_keys_for_eval
            )
        finally:
            self._restore_python_gc_after_training()
            self._close_profile_csv()

    def _ensure_python_gc_disabled_for_training(self) -> None:
        if self._python_gc_disabled_by_trainer or not getattr(self.args, "disable_python_gc", True):
            return

        self._python_gc_was_enabled = gc.isenabled()
        if self._python_gc_was_enabled:
            gc.disable()
        self._python_gc_disabled_by_trainer = True
        logger.info(
            "Python automatic GC disabled in the main training process; manual gc.collect() "
            "and accelerator cache flush will run every %s global steps.",
            getattr(self.args, "torch_empty_cache_steps", None),
        )

    def _restore_python_gc_after_training(self) -> None:
        if not self._python_gc_disabled_by_trainer:
            return

        collected = gc.collect()
        if self._python_gc_was_enabled and not gc.isenabled():
            gc.enable()
        logger.info(
            "Python automatic GC restored after training; final gc.collect() reclaimed %s objects.",
            collected,
        )

    def _maybe_run_manual_memory_cleanup(self) -> None:
        cleanup_steps = getattr(self.args, "torch_empty_cache_steps", None)
        if cleanup_steps is None:
            return

        cleanup_steps = int(cleanup_steps)
        if cleanup_steps <= 0:
            return

        step = int(self.state.global_step)
        if step <= 0 or step % cleanup_steps != 0 or step == self._manual_memory_cleanup_last_step:
            return

        collected = gc.collect()
        empty_cache()
        self._manual_memory_cleanup_last_step = step
        logger.info(
            "Manual memory cleanup at global_step=%s: gc.collect() reclaimed %s objects; "
            "accelerator empty_cache() called=%s.",
            step,
            collected,
            is_available("xpu") or is_available("cuda"),
        )

    def _update_loss_tracker_from_outputs(self, outputs: Union[Dict, Any]):
        """Extract and accumulate each sub-loss from the model's output dict."""
        for loss_type, loss_value in outputs.items():
            if "loss" in loss_type and loss_value is not None:
                # detach only, never .item() — the latter forces a CUDA sync.
                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.detach()

                stats = self.loss_tracker.setdefault(loss_type, {"total": 0.0, "count": 0})
                # Keep tensors as tensors while accumulating.
                if isinstance(loss_value, torch.Tensor):
                    if isinstance(stats["total"], float):
                        stats["total"] = torch.tensor(0.0, device=loss_value.device)
                    stats["total"] += loss_value
                else:
                    stats["total"] += float(loss_value)
                stats["count"] += 1

    def _scale_loss_for_backward(self, loss: torch.Tensor, num_items_in_batch=None) -> torch.Tensor:
        """Normalize the loss before backward exactly as HF Trainer does."""
        if self.args.n_gpu > 1:
            loss = loss.mean()

        if getattr(self, "use_apex", False):
            return loss

        if (not getattr(self, "model_accepts_loss_kwargs", False) or num_items_in_batch is None) and getattr(
            self, "compute_loss_func", None
        ) is None:
            loss = loss / getattr(
                self,
                "current_gradient_accumulation_steps",
                self.args.gradient_accumulation_steps,
            )

        return loss

    def _backward_loss(self, loss: torch.Tensor) -> None:
        """Route the backward pass through apex, deepspeed, or the accelerator."""
        if getattr(self, "use_apex", False):
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            return

        kwargs = {}
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            kwargs["scale_wrt_gas"] = False

        self.accelerator.backward(loss, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Run one training step.

        ``inputs`` is a single batch dict, taken from the training DataLoader's
        iterator by :class:`DummyDataLoader`. The execution skeleton below —
        cp_context, _prepare_inputs, compute_loss, scaling, backward — is the only
        ordering that has ever been validated; profiling points sit between the
        stages.
        """
        self._ensure_python_gc_disabled_for_training()
        self._ensure_step_profiling_hooks()

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        self._maybe_profile_cuda_sync()
        step_t0 = time.perf_counter()

        batch = inputs
        if isinstance(batch, dict):
            profile_batch_keys = [key for key in batch.keys() if key.startswith("profile_batch_")]
            for profile_key in profile_batch_keys:
                value = batch.pop(profile_key)
                metric_name = f"{_ROUTE_NAME}_{profile_key.removeprefix('profile_batch_')}"
                self._record_profile_average(metric_name, value)

        cp_context, batch = self._prepare_context_parallel_inputs(model, batch)

        with cp_context():
            self._maybe_profile_cuda_sync()
            to_device_t0 = time.perf_counter()
            batch = self._prepare_inputs(batch)
            self._maybe_profile_cuda_sync()
            self._record_profile_time("to_device_time_ms", (time.perf_counter() - to_device_t0) * 1000.0)

            self._maybe_profile_cuda_sync()
            forward_t0 = time.perf_counter()
            with self.compute_loss_context_manager():
                if self._is_step_profiling_enabled():
                    loss, model_outputs = self.compute_loss(
                        model,
                        batch,
                        return_outputs=True,
                        num_items_in_batch=num_items_in_batch,
                    )
                    self._record_model_profile_outputs(model_outputs)
                else:
                    loss = self.compute_loss(
                        model,
                        batch,
                        return_outputs=False,
                        num_items_in_batch=num_items_in_batch,
                    )
            self._maybe_profile_cuda_sync()
            self._record_profile_time("forward_time_ms", (time.perf_counter() - forward_t0) * 1000.0)

            del batch

            loss = self._scale_loss_for_backward(loss, num_items_in_batch=num_items_in_batch)
            self._maybe_profile_cuda_sync()
            backward_t0 = time.perf_counter()
            self._backward_loss(loss)
            self._maybe_profile_cuda_sync()
            self._record_profile_time("backward_time_ms", (time.perf_counter() - backward_t0) * 1000.0)
            total_loss = loss.detach()

        self._maybe_profile_cuda_sync()
        self._record_profile_time("step_time_ms", (time.perf_counter() - step_t0) * 1000.0)
        self._finish_profile_csv_step()

        return total_loss

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        loss_scale=1.0,
    ):
        """Compute the training loss, preferring the injected ``forward_fn``.

        When ``forward_fn`` is not ``None`` it computes this batch's loss;
        otherwise this falls back to the standard HF Trainer ``compute_loss``.

        Args:
            model: The model being trained.
            inputs: The input dict, already moved to the device.
            return_outputs: Whether to return the model outputs alongside the loss,
                for recording detailed metrics during evaluation.
            num_items_in_batch: Optional sample count used to normalize the loss.
            loss_scale: Scaling factor applied to the total loss.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
                - The loss alone when ``return_outputs=False``.
                - A ``(loss, outputs)`` tuple when ``return_outputs=True``.
        """
        # Use the custom forward function.
        if self.forward_fn is not None:
            res = self.forward_fn(model=model, inputs=inputs, loss_scale=loss_scale)

            if isinstance(res, tuple) and len(res) == 2:
                loss, outputs = res
            else:
                loss, outputs = res, {"loss": res}

            self._update_loss_tracker_from_outputs(outputs)
            # NOTE: the custom forward already computed the correct loss; HF's
            # cross-device scaling must not be applied on top of it.

            return (loss, outputs) if return_outputs else loss

        else:
            # NOTE: pass num_items_in_batch=None explicitly. Otherwise HF Trainer
            # multiplies the loss by num_processes when
            # average_tokens_across_devices=True, making the multi-GPU loss N times
            # the single-GPU value.
            if return_outputs:
                loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=None)
                self._update_loss_tracker_from_outputs(outputs)
                return (loss, outputs)
            else:
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch=None)

    def get_train_dataloader(self):
        """Return the training DataLoader.

        ``train.py`` has already built and passed in the DataLoader, so this only
        has to: run accelerator.prepare (unless physically_split), build the
        iterator or fast-forward it on resume, issue the NCCL warmup barrier, and
        return a DummyDataLoader for HF's step counting and progress bar.
        """
        if self.train_dataloader is None:
            return self._maybe_wrap_dataloader_for_profiling(super().get_train_dataloader())

        # 1. prepare dataloader
        if not hasattr(self, "_dataloader_prepared"):
            bs = self.train_dataloader.batch_size
            if bs is None and hasattr(self.train_dataloader.batch_sampler, "batch_size"):
                bs = self.train_dataloader.batch_sampler.batch_size
            self._train_dataloader_batch_size = bs
            ds = self.train_dataset
            if getattr(ds, "physically_split", False):
                # Data already sharded across ranks; skip accelerator.prepare()
                # to avoid double-sharding via DistributedSampler.
                logger.info(
                    "Train dataset is physically split -- skipping accelerator.prepare() "
                    f"(manual_sampler={getattr(ds, 'use_manual_distributed_sampler', False)}, "
                    f"physical_rank={getattr(ds, 'physical_shard_rank', 'n/a')}/"
                    f"{getattr(ds, 'physical_shard_world_size', 'n/a')}, "
                    f"logical_rank={getattr(ds, 'logical_shard_rank', 'n/a')}/"
                    f"{getattr(ds, 'logical_shard_world_size', 'n/a')})"
                )
            else:
                self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
                logger.info("Train dataloader prepared by accelerator")
            self._dataloader_prepared = True
            self._log_dataloader_summary()

        # 2. Build the iterator. On resume, _restore_data_state_if_needed builds it
        #    through skip_first_batches (fast-forwarding as it goes); otherwise it
        #    is built here.
        #    Never call iter(dl) eagerly on the resume path: it immediately spawns
        #    num_workers persistent workers and starts prefetching, and the
        #    subsequent skip_first_batches rebuild spawns a second full set. The
        #    original DataLoader still holds the first set, so those workers never
        #    exit and their prefetch queues stay full — per-node worker memory
        #    doubles. On a 96-node resume every node was SIGKILLed by the host OOM
        #    killer within seconds of "[Resume] skipped N batches".
        self._restore_data_state_if_needed()
        if not self._train_iterator_initialized and not self._resume_checkpoint_path:
            self._init_train_iterator()

        # 3. NCCL warmup barrier. The DataLoader workers are up and prefetching in
        #    the background, so the main process issues dist.barrier() here to
        #    trigger NCCL channel setup. The two overlap: workers fetch data while
        #    the main process initializes NCCL. Once the barrier returns, the first
        #    training step waits for neither, which measurably shortens it.
        if dist.is_available() and dist.is_initialized():
            logger.info(
                "[NCCL warmup] dist.barrier() start — NCCL channels initializing "
                "while DataLoader workers prefetch in background..."
            )
            import time as _time

            _t0 = _time.time()
            dist.barrier()
            logger.info(f"[NCCL warmup] dist.barrier() done in {_time.time() - _t0:.1f}s")

        # Return a dummy dataloader, used only for step counting and the progress bar.
        dataloader = DummyDataLoader(total_steps=self._calculate_total_training_steps(), trainer=self)
        return self._maybe_wrap_dataloader_for_profiling(dataloader)

    def num_examples(self, dataloader) -> int:
        # DummyDataLoader has no .dataset, so the base class would fall back to
        # estimating len(dl) * batch_size. We hold the real dataset — report its
        # actual sample count instead.
        if self.train_dataset is not None:
            return len(self.train_dataset)
        return super().num_examples(dataloader)

    def get_total_train_batch_size(self, args) -> int:
        batch_size = getattr(self, "_train_dataloader_batch_size", None)
        if batch_size is not None:
            dp_world_size = args.world_size // self.get_tp_size()
            total = batch_size * args.gradient_accumulation_steps * dp_world_size
            logger.info("Train batch_size=%d, total_train_batch_size=%d", batch_size, int(total))
            return int(total)
        return super().get_total_train_batch_size(args)

    def _calculate_total_training_steps(self) -> int:
        if self.args.max_steps > 0:
            return self.args.max_steps

        steps_per_epoch = len(self.train_dataloader)

        logger.warning(
            "max_steps not set, falling back to steps_per_epoch=%d. Recommend setting max_steps explicitly.",
            steps_per_epoch,
        )
        return steps_per_epoch

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        should_sync = (
            dist.is_available()
            and dist.is_initialized()
            and getattr(self.control, "should_save", False)
        )

        if should_sync:
            dist.barrier()

        result = super()._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate
        )

        if should_sync:
            dist.barrier()

        self._maybe_run_manual_memory_cleanup()

        if should_sync:
            dist.barrier()

        return result

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Inject the per-component losses into the log dict, then defer to the base class.

        Once every ``logging_steps``, computes and records the interval mean of
        each sub-loss (``vla_loss``), attaches the step-profile timing means when
        profiling is on, and resets the counters.

        Args:
            logs: The dict about to be written to the log; this method injects the
                per-component loss metrics into it.
            start_time: Training start timestamp, passed straight through to the
                base class.
        """
        if self._is_step_profiling_enabled():
            for key in sorted(getattr(self, "_profile_stats", {}).keys()):
                stats = self._profile_stats[key]
                if stats and stats["count"] > 0:
                    logs[key] = round(stats["total"] / stats["count"], 3)
                    stats["total"] = 0.0
                    stats["count"] = 0
            for key in sorted(getattr(self, "_profile_count_stats", {}).keys()):
                value = self._profile_count_stats[key]
                if value > 0:
                    logs[key] = int(value)
                    self._profile_count_stats[key] = 0.0

        # Average the losses: stack them into one tensor and all-reduce once,
        # rather than all-reducing and .item()-ing each separately, which would
        # cost N round trips of latency plus N device syncs.
        if self.loss_tracker:
            device = self.args.device
            totals_list: list[torch.Tensor] = []
            counts_list: list[float] = []
            for loss_type in _SYNCED_LOSS_LOG_KEYS:
                stats = self.loss_tracker.get(loss_type, {"total": 0.0, "count": 0})
                total = stats["total"]
                if isinstance(total, torch.Tensor):
                    t = total.detach().to(device=device, dtype=torch.float32)
                else:
                    t = torch.tensor(float(total), device=device, dtype=torch.float32)
                totals_list.append(t)
                counts_list.append(float(stats["count"]))

            totals_tensor = torch.stack(totals_list)
            counts_tensor = torch.tensor(counts_list, device=device, dtype=torch.float32)

            if (
                self._log_metrics_sync_across_ranks
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                totals_tensor = self.accelerator.reduce(totals_tensor, reduction="sum")
                counts_tensor = self.accelerator.reduce(counts_tensor, reduction="sum")

            totals_cpu = totals_tensor.cpu().tolist()
            counts_cpu = counts_tensor.cpu().tolist()

            for i, loss_type in enumerate(_SYNCED_LOSS_LOG_KEYS):
                if counts_cpu[i] > 0:
                    logs[loss_type] = round(totals_cpu[i] / counts_cpu[i], 6)
                if loss_type in self.loss_tracker:
                    self.loss_tracker[loss_type]["total"] = 0.0
                    self.loss_tracker[loss_type]["count"] = 0

        if self.train_dataloader is not None:
            if self.args.max_steps and self.args.max_steps > 0:
                # HF's built-in `epoch` is only a local DummyDataLoader progress value
                # and does not resume meaningfully. Override state.epoch itself so
                # Trainer.log() will emit the cumulative global training progress instead.
                self.state.epoch = round(self.state.global_step / self.args.max_steps, 6)
                logs["epoch"] = self.state.epoch

            dl_len = max(len(self.train_dataloader), 1)
            fractional_epoch = float(self._data_epoch) + float(self._steps_in_epoch) / float(dl_len)
            epochs_tensor = torch.tensor([fractional_epoch], device=self.args.device, dtype=torch.float32)
            if (
                self._log_metrics_sync_across_ranks
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                epochs_tensor = self.accelerator.reduce(epochs_tensor, reduction="mean")
            logs[f"{_ROUTE_NAME}_epoch"] = round(float(epochs_tensor.cpu().tolist()[0]), 6)

        return super().log(logs, start_time)

    def _save_checkpoint(self, model, trial) -> None:
        # The profiling monkey-patches on optimizer/lr_scheduler.step are
        # unpicklable local closures; drop them while state_dicts are saved.
        self._unwrap_step_profiling_hooks()
        try:
            super()._save_checkpoint(model, trial)
        finally:
            self._ensure_step_profiling_hooks()
        if self.train_dataloader is None:
            return

        checkpoint_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        state_path = self._get_data_state_path(checkpoint_dir)
        torch.save(self._data_state_dict(), state_path)

    def create_optimizer(self):
        """Create the optimizer, supporting differential multimodal learning rates."""
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]
                if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                    vision_tower_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name]
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n not in projector_parameters
                                    and n not in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n not in projector_parameters
                                    and n in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.vision_tower_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n not in projector_parameters
                                    and n not in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n not in projector_parameters
                                    and n in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.vision_tower_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
                else:
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            vla_action_head_lr = getattr(self.args, "vla_action_head_lr", None)
            if vla_action_head_lr is not None and vla_action_head_lr != 0:
                action_head_keywords = {
                    "dit",
                    "action_encoder",
                    "action_decoder",
                    "future_tokens",
                    "state_encoder",
                    "position_embedding",
                }
                action_head_param_ids = set()
                for n, p in opt_model.named_parameters():
                    if p.requires_grad and any(kw in n for kw in action_head_keywords):
                        action_head_param_ids.add(id(p))

                if action_head_param_ids:
                    new_groups = []
                    for group in optimizer_grouped_parameters:
                        remaining = [p for p in group["params"] if id(p) not in action_head_param_ids]
                        ah_params = [p for p in group["params"] if id(p) in action_head_param_ids]
                        if remaining:
                            new_groups.append({**group, "params": remaining})
                        if ah_params:
                            ah_group = {**group, "params": ah_params, "lr": vla_action_head_lr}
                            new_groups.append(ah_group)
                    optimizer_grouped_parameters = new_groups

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer


class DummyDataLoader:
    """Provide HF Trainer a compatible iteration interface for step counting and the progress bar.

    ``__next__`` calls ``trainer.get_next_batch()``, taking the next batch from the
    training DataLoader iterator the trainer holds. When that iterator is
    exhausted the trainer resets it and advances the epoch counter, so the total
    iteration length here depends only on ``total_steps``.
    """

    def __init__(self, total_steps: int, trainer: "Tau0VLATrainer"):
        self.total_steps = total_steps
        # weakref, to avoid a reference cycle.
        import weakref

        self._trainer_ref = weakref.ref(trainer)
        self.current_step = 0

    def __iter__(self):
        self.current_step = 0
        return self

    def __next__(self):
        if self.current_step >= self.total_steps:
            raise StopIteration

        self.current_step += 1
        trainer = self._trainer_ref()
        if trainer is None:
            raise RuntimeError("Trainer has been garbage collected")
        return trainer.get_next_batch()

    def __len__(self) -> int:
        return self.total_steps

    def cleanup(self):
        self._trainer_ref = None


class ProfilingDataLoaderProxy:
    """Wrap a dataloader-like object and record time spent waiting for ``next()``.

    The proxy is intentionally minimal so it can wrap both the custom
    ``DummyDataLoader`` and standard PyTorch/HF dataloaders without changing
    their public behavior.
    """

    def __init__(self, dataloader: Any, trainer: "Tau0VLATrainer"):
        import weakref

        self._dataloader = dataloader
        self._trainer_ref = weakref.ref(trainer)
        self._iterator = None

    def __iter__(self):
        self._iterator = iter(self._dataloader)
        return self

    def __next__(self):
        trainer = self._trainer_ref()
        if self._iterator is None:
            self._iterator = iter(self._dataloader)

        t0 = time.perf_counter()
        batch = next(self._iterator)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if trainer is not None:
            # With only one route, its per-route data_time is the global
            # data_time; there is nothing to break out separately.
            trainer._record_profile_time("data_time_ms", elapsed_ms)
        return batch

    def __len__(self):
        return len(self._dataloader)

    def __getattr__(self, name: str):
        return getattr(self._dataloader, name)


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Model-saving helper that understands DeepSpeed."""
    if trainer.deepspeed:
        synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class SaveProcessorCallback(TrainerCallback):
    """Training callback that saves the Processor alongside every checkpoint."""

    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if state.is_world_process_zero and self.processor is not None:
            self.processor.save_pretrained(ckpt_dir)


class FinchDataSpecCallback(TrainerCallback):
    """Mirror the Finch data contract into every ``checkpoint-N/`` directory.

    Background: ``FinchVLADataset.save_data_spec`` + ``save_policy_manifest``
    are called once during ``train()`` initialization to write
    ``finch_data_spec/`` + ``policy_manifest.json`` into the *output_dir*.
    HF Trainer's native save event, however, serializes model weights +
    processor into a child ``checkpoint-<step>/`` subdir. When deployment
    later asks ``Tau0VLAPolicy.from_checkpoint(ckpt_dir=<...>/checkpoint-N)``,
    that subdir needs its own copies of these artifacts — otherwise deploy
    has to walk up and hunt, which is a distribution-time footgun.

    This callback mirrors the small file artifacts as real copies, but
    ``finch_data_spec/`` as a RELATIVE SYMLINK to ``../finch_data_spec``:
    the spec tree holds one subdir per mix leaf, and a run dir that had
    accumulated stale inline-config dirs (294k observed) made the previous
    ``shutil.copytree`` mirror stall every checkpoint ~70 min with all ranks
    blocked on the save barrier. A symlink is O(1), always current, and
    ``tau0_vla.data.load_checkpoint_spec(checkpoint-N)`` /
    ``load_data_spec(checkpoint-N)`` resolve through it transparently.
    Caveat: copying a ``checkpoint-N`` elsewhere must either follow symlinks
    (``cp -rL`` / rsync -L) or also bring ``output_dir/finch_data_spec``.
    Rank-0 only; pure file I/O, no distributed coordination needed.
    """

    _COPY_ARTIFACTS = (
        "policy_manifest.json",
        "resolved_config_full.yaml",
        "run_spec.json",
    )
    _SYMLINK_ARTIFACTS = ("finch_data_spec",)

    def on_save(self, args, state, control, **kwargs):
        import shutil

        if not state.is_world_process_zero:
            return
        output_dir = os.path.abspath(args.output_dir)
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            return

        for name in self._COPY_ARTIFACTS:
            src = os.path.join(output_dir, name)
            dst = os.path.join(ckpt_dir, name)
            if not os.path.exists(src):
                continue
            if os.path.exists(dst):
                # Treat the per-checkpoint mirror as rebuilt-on-demand so a
                # mid-training rewrite (rare but possible) always propagates —
                # these artifacts are small single files.
                os.remove(dst)
            shutil.copy2(src, dst)

        for name in self._SYMLINK_ARTIFACTS:
            src = os.path.join(output_dir, name)
            dst = os.path.join(ckpt_dir, name)
            if not os.path.exists(src):
                continue
            if os.path.islink(dst):
                os.remove(dst)
            elif os.path.isdir(dst):
                # A real directory here is a full mirror from the copytree era
                # (possibly partial if that copy was interrupted). Leave it —
                # rmtree of a huge stale tree is the very stall this symlink
                # replaces, and deploy can read it as-is.
                logger.warning(
                    "FinchDataSpecCallback: %s already holds a copied directory; leaving it in place", dst
                )
                continue
            try:
                os.symlink(os.path.join("..", name), dst)
            except OSError as exc:
                logger.warning(
                    "FinchDataSpecCallback: could not symlink %s -> ../%s (%s); "
                    "deploy must read the spec from the run dir instead",
                    dst,
                    name,
                    exc,
                )
