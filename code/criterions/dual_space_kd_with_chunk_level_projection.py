import math
import torch
from .dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA


class DualSpaceKDWithCLP(DualSpaceKDWithCMA):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)

    def compute_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, input_data, output_data, distiller, log, chunk_mask=None
    ):
        target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        
        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

        if hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "model") \
            and hasattr(distiller.teacher_model.model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "transformer") \
            and hasattr(distiller.teacher_model.model, "wte"):
            tea_embed_tokens = distiller.teacher_model.transformer.wte
        else:
            raise NotImplementedError 

        formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach()

        norm_tea_target_embeds = tea_target_embeds / tea_target_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        stu_v_hiddens = distiller.projectors["s2t"](hiddens).float()
        tea_v_hiddens = distiller.projectors["t2s"](
            norm_teacher_hiddens + norm_tea_target_embeds
        ).float()

        if chunk_mask is None:
            stu_offsets = input_data["offsets"]
            tea_offsets = input_data[f"teacher_{distiller.teacher_model_type}_offsets"]
            chunk_mask_shape = (hiddens.shape[0], hiddens.shape[1], teacher_hiddens.shape[1])
            chunk_mask = torch.zeros(chunk_mask_shape, dtype=torch.float32, device=hiddens.device)
            chunk_mask, _, _, _ = self.get_chunk_alignment_mask(stu_offsets, tea_offsets, chunk_mask)
        else:
            chunk_mask = chunk_mask.float()

        chunk_mask = chunk_mask * pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
        t2s_weight = chunk_mask / chunk_mask.sum(dim=2, keepdim=True).clamp(min=1e-10)    
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens).to(hiddens)
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        ) # Equation 4 (except where is the softmax?)
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0] # Equation 5
        t2s_acc_mask = t2s_logits.argmax(-1).eq(target)
        t2s_acc = (t2s_acc_mask * pad_mask).sum()
        max_probs = (t2s_logits.softmax(-1).max(-1)[0] * pad_mask).sum()
        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_acc"] = t2s_acc
        log["max_t2s_prob"] = max_probs
        
        if not self.args.only_save_projector:  # skip if only train projectors (pre-train projectors)
            t2s_kd_loss = self.dist_func(
                outputs.logits, t2s_logits.detach(), target, reduction="none", use_tea_temp=True
            ) # Equation 6
            t2s_kd_loss = (t2s_kd_loss * pad_mask * t2s_acc_mask).sum() 

            teacher_chunk_mask = chunk_mask.transpose(-1, -2)
            s2t_weight = teacher_chunk_mask / teacher_chunk_mask.sum(dim=2, keepdim=True).clamp(min=1e-10)
            s2t_hiddens = s2t_weight.matmul(stu_v_hiddens).to(hiddens)
            s2t_logits = s2t_hiddens.matmul(
                distiller.teacher_model.lm_head.weight.detach().transpose(-1, -2)
            ) # Equation 8 (except where is the softmax?)

            s2t_kd_loss = self.compute_forward_kl_divergence(
                s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
            ) # Equation 9
            s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum()
            s2t_acc = (s2t_logits.argmax(-1).eq(teacher_target) * teacher_pad_mask).sum() * pad_mask.sum() / teacher_pad_mask.sum()

            kd_loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss
            log["t2s_kd_loss"] = t2s_kd_loss
            log["s2t_kd_loss"] = s2t_kd_loss
            log["s2t_acc"] = s2t_acc
        else:
            kd_loss = t2s_ce_loss

        log["kd_loss"] = kd_loss
        return kd_loss, log, t2s_weight, None, None
    