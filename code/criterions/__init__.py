from .cross_entropy_loss import CrossEntropyLoss
from .various_divergence import VariousDivergence
from .dual_space_kd import DualSpaceKD
from .dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA
from .dual_space_kd_with_chunk_level_projection import DualSpaceKDWithCLP
from .dual_space_kd_with_chunk_level_attention import DualSpaceKDWithCLA
from .dual_space_kd_with_cma_plus_kq_matching import DualSpaceKDWithCMAPlusKQMatching
from .universal_logit_distillation import UniversalLogitDistillation
from .min_edit_dis_kld import MinEditDisForwardKLD


criterion_list = {
    "cross_entropy": CrossEntropyLoss,
    "various_divergence": VariousDivergence,
    "dual_space_kd": DualSpaceKD,
    "dual_space_kd_with_cma": DualSpaceKDWithCMA,
    "dual_space_kd_with_clp": DualSpaceKDWithCLP,
    "dual_space_kd_with_cla": DualSpaceKDWithCLA,
    "dual_space_kd_with_cma_plus_kq_matching" : DualSpaceKDWithCMAPlusKQMatching,
    "universal_logit_distillation": UniversalLogitDistillation,
    "min_edit_dis_kld": MinEditDisForwardKLD,
}

def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    else:
        raise NameError(f"Undefined criterion for {args.criterion}!")