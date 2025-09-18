import math
import torch
from .cross_entropy_loss import CrossEntropyLoss


class VariousDivergence(CrossEntropyLoss):
    def __init__(self, args, padding_id=-100) -> None:
        super(VariousDivergence, self).__init__(args, padding_id=padding_id)
        self.kd_rate = args.kd_rate
        self.kd_temp = args.kd_temperature
        self.tea_temp = args.teacher_temperature
        self.kd_objective = args.kd_objective
        self.args = args

        if self.kd_objective == "forward_kl":
            self.dist_func = self.compute_forward_kl_divergence
        elif self.kd_objective == "reverse_kl":
            self.dist_func = self.compute_reverse_kl_divergence
        elif self.kd_objective == "adaptive_kl":
            self.dist_func = self.compute_adaptive_kl_divergence
        elif self.kd_objective == "skewed_forward_kl":
            self.dist_func = self.compute_skewed_forward_kl_divergence
        elif self.kd_objective == "skewed_reverse_kl":
            self.dist_func = self.compute_skewed_reverse_kl_divergence
        elif self.kd_objective == "js_divergence":
            self.dist_func = self.compute_js_divergence
        else:
            raise NameError(f"Unsupported kd_objective for `{self.kd_objective}'")
    
    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        self.distiller = distiller
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None), 
            output_hidden_states=True
        )
        logits = outputs.logits
        log = {}
        loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0]

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True)
            teacher_logits = teacher_outputs.logits
        
        # Qwen has different vocab_size for models in different sizes (see https://github.com/QwenLM/Qwen/issues/419)
        if self.args.model_type == "qwen":
            logits = logits[..., :151851]
            teacher_logits = teacher_logits[..., :151851]
        
        kd_loss = self.dist_func(logits, teacher_logits, output_data["label"])
        log["kd_loss"] = kd_loss

        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(logits, output_data["label"])
        log["accuracy"] = accuracy

        if self.args.report_logits:
            self.record_logits(
                logits, 
                output_data["label"], 
                log, 
                teacher_logits=teacher_logits, 
                teacher_target=output_data[f"teacher_{distiller.teacher_model_type}_label"]
            )

        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return {"loss":loss / batch_denom, "logits":logits, "log":logging_output}
    
    
    def get_chunk_alignment_mask(self, s_offsets, t_offsets, chunk_mask):
        # Group (chunk number) assignment for each token
        s_groups = torch.zeros((chunk_mask.shape[0], chunk_mask.shape[1]), dtype=torch.int64, device=chunk_mask.device)
        t_groups = torch.zeros((chunk_mask.shape[0], chunk_mask.shape[2]), dtype=torch.int64, device=chunk_mask.device)
        max_group = 0

        # For each sample
        for i in range(len(s_offsets)):
            group = 0
            s_chunk_end, t_chunk_end = 0, 0
            while s_chunk_end < len(s_offsets[i]) and t_chunk_end < len(t_offsets[i]):
                # Start new chunk
                s_chunk_start, t_chunk_start = s_chunk_end, t_chunk_end
                s_token_end, t_token_end = s_offsets[i, s_chunk_end], t_offsets[i, t_chunk_end]

                # Find end token of current chunk
                while s_token_end != t_token_end:
                    if s_token_end < t_token_end:
                        # Moving from prompt to response
                        if (s_chunk_end == len(s_offsets[i]) - 1) or (s_offsets[i, s_chunk_end + 1] < s_token_end):
                            break
                        s_chunk_end += 1 
                        s_token_end = s_offsets[i, s_chunk_end]
                    else:
                        # Moving from prompt to response
                        if (t_chunk_end == len(t_offsets[i]) - 1) or (t_offsets[i, t_chunk_end + 1] < t_token_end):
                            break
                        t_chunk_end += 1
                        t_token_end = t_offsets[i, t_chunk_end]

                # Update mask and group assignment
                chunk_mask[i, s_chunk_start : s_chunk_end + 1, t_chunk_start : t_chunk_end + 1] = 1.
                s_groups[i, s_chunk_start : s_chunk_end + 1] = group
                t_groups[i, t_chunk_start : t_chunk_end + 1] = group

                # First token of each chunk made negative (to mark chunk beginnings)
                s_groups[i, s_chunk_start] = -group
                t_groups[i, t_chunk_start] = -group

                s_chunk_end, t_chunk_end = s_chunk_end + 1, t_chunk_end + 1
                group += 1

            if group > max_group:
                max_group = group

        return chunk_mask, s_groups, t_groups, max_group
    

    def apply_temperature_scaling(self, s_logits, t_logits, use_tea_temp=False):
        s_logits = s_logits / self.kd_temp
        t_logits = t_logits / self.kd_temp
        t_logits = t_logits / self.tea_temp if use_tea_temp else t_logits
        return s_logits, t_logits
    

    def compute_forward_kl_divergence(
        self, 
        logits, 
        teacher_logits, 
        target, 
        reduction="sum", 
        log=None, 
        use_tea_temp=False
    ):
        logits, teacher_logits = self.apply_temperature_scaling(logits, teacher_logits, use_tea_temp=use_tea_temp)

        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (teacher_probs * (teacher_lprobs - lprobs)) # definition of KL divergence
        inf_mask = logits.isinf()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)
        
        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["forward_kl"] = kld

        return kld
    
    def compute_reverse_kl_divergence(
        self, 
        logits, 
        teacher_logits, 
        target, 
        reduction="sum", 
        log=None, 
        use_tea_temp=False
    ):
        logits, teacher_logits = self.apply_temperature_scaling(logits, teacher_logits, use_tea_temp=use_tea_temp)

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (probs * (lprobs - teacher_lprobs))
        inf_mask = logits.isinf() | teacher_logits.isinf()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)

        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["reverse_kl"] = kld

        return kld
    
    def compute_adaptive_kl_divergence(
        self, 
        logits, 
        teacher_logits, 
        target, 
        reduction="sum", 
        log=None, 
        use_tea_temp=False
    ):
        alpha = self.args.adaptive_kl_alpha

        logits, teacher_logits = self.apply_temperature_scaling(logits, teacher_logits, use_tea_temp=use_tea_temp)
        probs = torch.softmax(logits, dim=-1)
        teacher_probs = torch.softmax(teacher_logits, dim=-1)

        sorted_teacher_probs, sorted_idx = teacher_probs.sort(-1)
        sorted_probs = probs.gather(-1, sorted_idx)
        gap = (sorted_teacher_probs - sorted_probs).abs()
        cum_teacher_probs = torch.cumsum(sorted_teacher_probs, -1)
        tail_mask = cum_teacher_probs.le(alpha).float()
        g_head = (gap * (1 - tail_mask)).sum(-1).detach()
        g_tail = (gap * tail_mask).sum(-1).detach()

        fkl = self.compute_forward_kl_divergence(logits, teacher_logits, target, reduction="none", use_tea_temp=use_tea_temp)
        rkl = self.compute_reverse_kl_divergence(logits, teacher_logits, target, reduction="none", use_tea_temp=use_tea_temp)

        akl = (g_head / (g_head + g_tail)) * fkl + (g_tail / (g_head + g_tail)) * rkl
        
        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            akl = akl.masked_fill_(pad_mask, 0.0)
            akl = akl.sum()

            if log is not None:
                log["adaptive_kl"] = akl

        return akl
    
    def compute_skewed_forward_kl_divergence(
        self, 
        logits, 
        teacher_logits, 
        target, 
        reduction="sum", 
        log=None, 
        use_tea_temp=False
    ):
        logits, teacher_logits = self.apply_temperature_scaling(logits, teacher_logits, use_tea_temp=use_tea_temp)

        student_probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        mixed_probs = self.args.skew_lambda * teacher_probs + (1 - self.args.skew_lambda) * student_probs
        mixed_lprobs = torch.log(mixed_probs)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (teacher_probs * (teacher_lprobs - mixed_lprobs))
        inf_mask = logits.isinf() | teacher_logits.isinf()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)
        
        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["skewed_forward_kl"] = kld

        return kld
    
    def compute_skewed_reverse_kl_divergence(
        self, 
        logits, 
        teacher_logits, 
        target, 
        reduction="sum", 
        log=None, 
        use_tea_temp=False
    ):
        logits, teacher_logits = self.apply_temperature_scaling(logits, teacher_logits, use_tea_temp=use_tea_temp)

        student_probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        mixed_probs = (1 - self.args.skew_lambda) * teacher_probs + self.args.skew_lambda * student_probs
        mixed_lprobs = torch.log(mixed_probs)
        student_lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        # teacher_lprobs = torch.log_softmax(teacher_logits / self.tea_temp / self.kd_temp, -1, dtype=torch.float32)
        kld = (student_probs * (student_lprobs - mixed_lprobs))
        inf_mask = logits.isinf() | teacher_logits.isinf()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)
        
        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["skewed_reverse_kl"] = kld

        return kld

    def compute_js_divergence(
        self, 
        logits, 
        teacher_logits, 
        target, 
        reduction="sum", 
        log=None, 
        use_tea_temp=False
    ):
        logits, teacher_logits = self.apply_temperature_scaling(logits, teacher_logits, use_tea_temp=use_tea_temp)

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        m_probs = (probs + teacher_probs) / 2
        
        lprobs = torch.log(probs + 1e-9)
        teacher_lprobs = torch.log(teacher_probs + 1e-9)
        m_lprobs = torch.log(m_probs + 1e-9)

        kld1 = teacher_probs * (teacher_lprobs - m_lprobs)
        kld2 = probs * (lprobs - m_lprobs)
        kld = (kld1 + kld2) / 2
        inf_mask = logits.isinf() | teacher_logits.isinf()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)
        
        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["js_div"] = kld

        return kld
