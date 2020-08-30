from ray.rllib.agents.ppo.ppo import *
from ray.rllib.agents.ppo.ppo_torch_policy import *
from ray.rllib.evaluation.postprocessing import *


class CustomPPOLoss:
    def __init__(self,
                 dist_class,
                 model,
                 value_targets,
                 advantages,
                 actions,
                 prev_logits,
                 prev_actions_logp,
                 vf_preds,
                 curr_action_dist,
                 value_fn,
                 cur_kl_coeff,
                 valid_mask,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 use_gae=True,
                 rollback_alpha=0.01):

        if valid_mask is not None:
            num_valid = torch.sum(valid_mask)

            def reduce_mean_valid(t):
                return torch.sum(t[valid_mask]) / num_valid

        else:

            def reduce_mean_valid(t):
                return torch.mean(t)

        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        logp_ratio = torch.exp(curr_action_dist.logp(actions) - prev_actions_logp)
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)

        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)

        if rollback_alpha > 0:
            pos_adv_rollback = torch.where(
                logp_ratio <= 1.0 + clip_param,
                logp_ratio,
                -rollback_alpha * logp_ratio + (1.0 + rollback_alpha) * (1.0 + clip_param)
            )
            neg_adv_rollback = torch.where(
                logp_ratio >= 1.0 - clip_param,
                logp_ratio,
                -rollback_alpha * logp_ratio + (1.0 + rollback_alpha) * (1.0 - clip_param)
            )
            surrogate_loss = advantages * torch.where(
                advantages >= 0,
                pos_adv_rollback,
                neg_adv_rollback
            )
            self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)
        else:
            surrogate_loss = torch.min(advantages * logp_ratio,
                                       advantages * torch.clamp(logp_ratio, 1 - clip_param, 1 + clip_param))
            self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        if use_gae:
            vf_loss1 = torch.pow(value_fn - value_targets, 2.0)
            vf_clipped = vf_preds + torch.clamp(value_fn - vf_preds,
                                                -vf_clip_param, vf_clip_param)
            vf_loss2 = torch.pow(vf_clipped - value_targets, 2.0)
            vf_loss = torch.max(vf_loss1, vf_loss2)
            self.mean_vf_loss = reduce_mean_valid(vf_loss)
            loss = reduce_mean_valid(
                -surrogate_loss + cur_kl_coeff * action_kl +
                vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy)
        else:
            self.mean_vf_loss = 0.0
            loss = reduce_mean_valid(-surrogate_loss +
                                     cur_kl_coeff * action_kl -
                                     entropy_coeff * curr_entropy)
        self.loss = loss


def custom_ppo_surrogate_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch, is_training=True)
    action_dist = dist_class(logits, model)

    mask = None
    if state:
        max_seq_len = torch.max(train_batch["seq_lens"])
        mask = sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = torch.reshape(mask, [-1])

    policy.loss_obj = CustomPPOLoss(
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.ACTION_DIST_INPUTS],
        train_batch[SampleBatch.ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        model.value_function(),
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"]
    )

    return policy.loss_obj.loss


CustomPPOTorchPolicy = build_torch_policy(
    name="CustomPPOTorchPolicy",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=custom_ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_out_fn=vf_preds_fetches,
    postprocess_fn=postprocess_ppo_gae,
    extra_grad_process_fn=apply_grad_clipping,
    before_init=setup_config,
    after_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ])


def get_custom_policy_class(config):
    assert config["framework"] == "torch"
    return CustomPPOTorchPolicy


CustomPPOTrainer = build_trainer(
    name="CustomPPO",
    default_config=DEFAULT_CONFIG,
    default_policy=CustomPPOTorchPolicy,
    get_policy_class=get_custom_policy_class,
    execution_plan=execution_plan,
    validate_config=validate_config)
