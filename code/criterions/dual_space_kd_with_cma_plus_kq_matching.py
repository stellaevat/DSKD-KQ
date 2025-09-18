import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from .dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA


class DualSpaceKDWithCMAPlusKQMatching(DualSpaceKDWithCMA):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)
        self.kq_rate = args.kq_rate

    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
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
        
        kd_loss, log, cma, keys, queries = self.compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )

        kq_loss = self.compute_kq_matching_loss(keys, queries, distiller)
        log["kq_loss"] = kq_loss

        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss + self.kq_rate * kq_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(
            logits, output_data["label"], 
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return {"loss":loss / batch_denom, "logits":logits, "alignment":cma, "log":logging_output}
    

# ==========================================================================================================
# Adapted from: https://github.com/szhang42/alignment_attention
# ==========================================================================================================

    def compute_kq_matching_loss(
        self, k, q, distiller
    ):
        k_reverse = GradReverse.apply(k, 1)
        q_reverse = GradReverse.apply(q, 1)

        if distiller.kq_adver_type == 'gan':
            k_out_real = self.discriminator(k_reverse, distiller)
            k_target_real = Variable(torch.ones_like(k_out_real)).cuda().detach()
            d_loss_real = distiller.kq_criterion(k_out_real, k_target_real)
             
            q_out_fake = self.discriminator(q_reverse.detach(), distiller)
            q_target_fake = Variable(torch.zeros_like(q_out_fake)).cuda().detach()
            d_loss_fake = distiller.kq_criterion(q_out_fake, q_target_fake)

            d_loss = d_loss_real + d_loss_fake
            kq_loss = d_loss.mean()

        elif distiller.kq_adver_type == 'ct':
            k_out_real = self.critic1(k_reverse, distiller)
            q_out_fake = self.critic1(q_reverse, distiller)

            cost = self.fast_cdist(k_out_real, q_out_fake)

            k_out_real_head = self.critic2(k_reverse, distiller)
            q_out_fake_head = self.critic2(q_reverse, distiller)

            k_out_real_tran = k_out_real_head.transpose(1, 2).contiguous()
            q_out_fake_tran = q_out_fake_head.transpose(1, 2).contiguous()

            cost_head = self.fast_cdist(k_out_real_tran, q_out_fake_tran)

            n_x = self.navigator1(k_reverse, distiller)
            n_y = self.navigator1(q_reverse, distiller)
            d = torch.matmul(n_x, n_y.transpose(-1, -2))

            n_x_tran = self.navigator2(k_reverse, distiller)
            n_y_tran = self.navigator2(q_reverse, distiller)

            n_x_tran = n_x_tran.transpose(1, 2)
            n_y_tran = n_y_tran.transpose(1, 2)
            d_head = torch.matmul(n_x_tran, n_y_tran.transpose(-1, -2))

            m_backward = F.softmax(d, dim=-2)  # backward transport map key
            m_forward = F.softmax(d, dim=-1)  # forward transport map query

            m_backward_head = F.softmax(d_head, dim=-2)  # backward transport map key
            m_forward_head = F.softmax(d_head, dim=-1)  # forward transport map query

            fw_rate = 0.5
            err = - ((1 - fw_rate) * (cost * m_backward).sum(-2).mean() + fw_rate * (cost * m_forward).sum(-1).mean())
            err_head = - ((1 - fw_rate) * (cost_head * m_backward_head).sum(-2).mean() + fw_rate * (
                        cost_head * m_forward_head).sum(-1).mean())
            
            err_loss = err + err_head
            kq_loss = err_loss.mean()

        return kq_loss
    
    def discriminator(self, x, distiller):
        highway = distiller.kq_highway1(x)
        pred = torch.sigmoid(highway) *  F.relu(highway) + (1. - torch.sigmoid(highway)) * x
        pred = distiller.kq_linear2(distiller.kq_nonlinear(distiller.kq_linear1(distiller.kq_dropout1(pred))))
        return pred

    def critic1(self, x, distiller, eps=1e-6):
        highway = distiller.kq_highway1(x)  # batch_size * num_filters_sum
        pred = torch.sigmoid(highway) *  F.relu(highway) + (1. - torch.sigmoid(highway)) * x
        pred = distiller.kq_linear4(distiller.kq_nonlinear(distiller.kq_linear3(distiller.kq_dropout1(pred))))
        pred = pred / (pred.norm(p=2, dim=-1, keepdim=True) + eps)
        return pred
    
    def critic2(self, x, distiller, eps=1e-6):
        highway = distiller.kq_highway2(x)
        pred = torch.sigmoid(highway) *  F.relu(highway) + (1. - torch.sigmoid(highway)) * x
        pred = distiller.kq_linear8(distiller.kq_nonlinear3(distiller.kq_linear7(distiller.kq_dropout2(pred))))
        pred = pred / (pred.norm(p=2, dim=-1, keepdim=True) + eps)
        return pred
    
    def navigator1(self, x, distiller, eps=1e-6):
        pred = distiller.kq_linear6(distiller.kq_nonlinear2(distiller.kq_linear5(x)))
        pred = pred / (pred.norm(p=2, dim=-1, keepdim=True) + eps)
        return pred

    def navigator2(self, x, distiller, eps=1e-6):
        pred = distiller.kq_linear10(distiller.kq_nonlinear4(distiller.kq_linear9(x)))
        pred = pred / (pred.norm(p=2, dim=-1, keepdim=True) + eps)
        return pred
    
    def fast_cdist(self, x1, x2):
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

        # Compute squared distance matrix using quadratic expansion
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))
        res.clamp_min_(1e-30).sqrt_()
        return res


class GradReverse(Function):
	@staticmethod
	def forward(ctx, x, beta):
		ctx.beta = beta
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		grad_input = grad_output.neg() * ctx.beta
		return grad_input, None
    
# ==========================================================================================================