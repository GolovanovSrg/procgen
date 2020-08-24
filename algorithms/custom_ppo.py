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


# def custom_postprocess_ppo_gae(policy,
#                         sample_batch,
#                         other_agent_batches=None,
#                         episode=None):

#     completed = sample_batch[SampleBatch.DONES][-1]
#     if completed:
#         last_r = 0.0
#     else:
#         next_state = []
#         for i in range(policy.num_state_tensors()):
#             next_state.append(sample_batch["state_out_{}".format(i)][-1])
#         last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
#                                sample_batch[SampleBatch.ACTIONS][-1],
#                                sample_batch[SampleBatch.REWARDS][-1],
#                                *next_state)
#     batch = custom_compute_advantages(
#         sample_batch,
#         last_r,
#         policy.config["gamma"],
#         policy.config["lambda"],
#         use_gae=policy.config["use_gae"])
#     return batch


# def custom_compute_advantages(rollout: SampleBatch,
#                        last_r: float,
#                        gamma: float = 0.9,
#                        lambda_: float = 1.0,
#                        use_gae: bool = True,
#                        use_critic: bool = True,
#                        alpha=0.1):

#     rollout_size = len(rollout[SampleBatch.ACTIONS])

#     assert SampleBatch.VF_PREDS in rollout or not use_critic, \
#         "use_critic=True but values not found"
#     assert use_critic or not use_gae, \
#         "Can't use gae without using a value function"

#     rewards = rollout[SampleBatch.REWARDS] + alpha * rollout[SampleBatch.ACTION_LOGP].clip(-1, 0)

#     if use_gae:
#         vpred_t = np.concatenate([rollout[SampleBatch.VF_PREDS], np.array([last_r])])
#         delta_t = (rewards + gamma * vpred_t[1:] - vpred_t[:-1])

#         rollout[Postprocessing.ADVANTAGES] = discount(delta_t, gamma * lambda_)
#         rollout[Postprocessing.VALUE_TARGETS] = (rollout[Postprocessing.ADVANTAGES] + rollout[SampleBatch.VF_PREDS]).copy().astype(np.float32)
#     else:
#         rewards_plus_v = np.concatenate([rewards, np.array([last_r])])
#         discounted_returns = discount(rewards_plus_v, gamma)[:-1].copy().astype(np.float32)

#         if use_critic:
#             rollout[Postprocessing.ADVANTAGES] = discounted_returns - rollout[SampleBatch.VF_PREDS]
#             rollout[Postprocessing.VALUE_TARGETS] = discounted_returns
#         else:
#             rollout[Postprocessing.ADVANTAGES] = discounted_returns
#             rollout[Postprocessing.VALUE_TARGETS] = np.zeros_like(rollout[Postprocessing.ADVANTAGES])

#     rollout[Postprocessing.ADVANTAGES] = rollout[Postprocessing.ADVANTAGES].copy().astype(np.float32)

#     assert all(val.shape[0] == rollout_size for key, val in rollout.items()), \
#         "Rollout stacked incorrectly!"
#     return rollout


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
