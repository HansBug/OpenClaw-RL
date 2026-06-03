import logging
import time

import ray
import wandb

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.misc import should_run_periodic_action

logger = logging.getLogger(__name__)


def _relay_pending_metrics(result):
    """Log pending wandb metrics relayed from secondary processes."""
    if not result:
        return
    if wandb.run is None:
        return
    metrics_list = result if isinstance(result, list) else [result]
    for item in metrics_list:
        if isinstance(item, list):
            for m in item:
                if isinstance(m, dict):
                    wandb.log(m)
        elif isinstance(item, dict):
            wandb.log(item)


def _get_rollout_generation_result(args, rollout_manager, rollout_id):
    max_retries = int(getattr(args, "rollout_generation_max_retries", 0) or 0)
    initial_backoff = max(0.0, float(getattr(args, "rollout_generation_retry_initial_backoff", 30.0) or 0.0))
    max_backoff = max(0.0, float(getattr(args, "rollout_generation_retry_max_backoff", 300.0) or 0.0))
    multiplier = max(1.0, float(getattr(args, "rollout_generation_retry_backoff_multiplier", 2.0) or 1.0))
    attempt = 0

    while True:
        try:
            return ray.get(rollout_manager.generate.remote(rollout_id))
        except Exception as exc:
            attempt += 1
            if max_retries >= 0 and attempt > max_retries:
                logger.error(
                    "Rollout generation failed permanently: rollout_id=%s attempts=%s max_retries=%s error=%r",
                    rollout_id,
                    attempt,
                    max_retries,
                    exc,
                )
                raise

            wait_s = initial_backoff * (multiplier ** max(0, attempt - 1))
            if max_backoff > 0:
                wait_s = min(wait_s, max_backoff)
            logger.warning(
                "Rollout generation failed; retrying same rollout after %.1fs: "
                "rollout_id=%s attempt=%s max_retries=%s error=%r",
                wait_s,
                rollout_id,
                attempt,
                "unlimited" if max_retries < 0 else max_retries,
                exc,
            )
            if wait_s > 0:
                time.sleep(wait_s)


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        _relay_pending_metrics(ray.get(rollout_manager.eval.remote(rollout_id=0)))

    def offload_train():
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()

    def save(rollout_id):
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            _relay_pending_metrics(ray.get(rollout_manager.eval.remote(rollout_id)))

        gen_result = _get_rollout_generation_result(args, rollout_manager, rollout_id)
        if isinstance(gen_result, tuple):
            rollout_data_ref, pending = gen_result
            _relay_pending_metrics(pending)
        else:
            rollout_data_ref = gen_result

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                _relay_pending_metrics(ray.get(actor_model.async_train(rollout_id, rollout_data_ref)))
            _relay_pending_metrics(ray.get(critic_train_handle))
        else:
            _relay_pending_metrics(ray.get(actor_model.async_train(rollout_id, rollout_data_ref)))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        offload_train()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            _relay_pending_metrics(ray.get(rollout_manager.eval.remote(rollout_id)))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
