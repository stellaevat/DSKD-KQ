import time
import os
import sys

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
import deepspeed

import json
from tqdm import tqdm

from arguments import get_args

from utils import initialize, save_rank, print_rank
from data_utils.distill_datasets import DistillDataset

from criterions import build_criterion
from distiller import Distiller
from distillation import prepare_dataset

torch.set_num_threads(1)
torch.set_printoptions(
    precision=10,
    threshold=sys.maxsize,
    linewidth=2**14,
    sci_mode=False
)


def run_model(args, distiller, dataset: DistillDataset, device):
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    criterion = build_criterion(args)
    
    sampler = DistributedSampler(
        dataset, 
        shuffle=False, 
        drop_last=False, 
        rank=dp_rank, 
        num_replicas=dp_world_size
    )
    dataloader = DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=args.eval_batch_size, 
        num_workers=args.num_workers, 
        collate_fn=dataset.collate
    )
    logging_output = {
        "epoch": 0,
        "global_step": 0,
        "loss": [], 
        "nll_loss": [],
        "kd_loss": [],
        "accuracy": [],
        "micro_step_time": [],
        "step_time": []
    }
    
    all_student_ids = []
    all_teacher_ids = []
    all_student_masks = []
    all_teacher_masks = []
    all_alignment_matrices = []

    distiller.eval()

    data_iter = iter(dataloader)
    num_samples = args.eval_num_samples if args.eval_num_samples else len(dataloader)
    num_batches = num_samples // args.eval_batch_size
    num_batches = num_batches + 1 if num_batches * args.eval_batch_size < num_samples else num_batches

    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc=f"Analysing {args.data_names} ", disable=(dist.get_rank() != 0)):
            input_batch, output_batch, _ = next(data_iter)
            dataset.move_to_device([input_batch, output_batch], device)
            
            batch = {}
            batch["input_batch"] = input_batch
            batch["output_batch"] = output_batch
            
            out_distill = distiller(criterion, batch, logging_output, 1)
            alignment_matrix = out_distill.get("alignment", None)
            all_alignment_matrices.append(alignment_matrix)

            student_ids = input_batch["input_ids"]
            teacher_ids = input_batch[f"teacher_{distiller.teacher_model_type}_input_ids"]
            all_student_ids.append(student_ids)
            all_teacher_ids.append(teacher_ids)
            
            student_target = output_batch["label"]
            teacher_target = output_batch[f"teacher_{distiller.teacher_model_type}_label"]
            student_mask = student_target.ne(criterion.padding_id)
            teacher_mask = teacher_target.ne(criterion.padding_id)
            all_student_masks.append(student_mask)
            all_teacher_masks.append(teacher_mask)
        
    all_student_ids = torch.cat(all_student_ids)
    all_teacher_ids = torch.cat(all_teacher_ids)
    all_student_masks = torch.cat(all_student_masks)
    all_teacher_masks = torch.cat(all_teacher_masks)
    all_alignment_matrices = torch.cat(all_alignment_matrices)
        
    return (
        all_student_ids,
        all_teacher_ids,
        all_student_masks,
        all_teacher_masks,
        all_alignment_matrices
        )


def analyse_main(args, distiller, dataset: DistillDataset, device):
    student_ids, teacher_ids, student_masks, teacher_masks, alignment_matrices = run_model(args, distiller, dataset, device)
    
    student_tokenizer = distiller.student_tokenizer
    teacher_tokenizer = distiller.teacher_tokenizers[distiller.teacher_model_type]
    student_tokens = [student_tokenizer.convert_ids_to_tokens(sequence) for sequence in student_ids]
    teacher_tokens = [teacher_tokenizer.convert_ids_to_tokens(sequence) for sequence in teacher_ids]

    jsonl_fname = os.path.join(args.save_dir, "align.jsonl")
    with open(jsonl_fname, "w") as f:
        for i, (s, t, s_mask, t_mask, align) in tqdm(enumerate(zip(student_tokens, teacher_tokens, student_masks, teacher_masks, alignment_matrices)), desc=f"Saving "):
            s_index = s_mask.nonzero().squeeze()
            t_index = t_mask.nonzero().squeeze()
            
            align_masked = align[s_index][:, t_index]
            s_masked = [s[idx] for idx in s_index]
            t_masked = [t[idx] for idx in t_index]

            jsonl_str = json.dumps({
                "student" : s_masked,
                "teacher" : t_masked,
                "alignment" : align_masked.tolist()
            })
            f.write(jsonl_str + "\n")

@record
def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    device = torch.cuda.current_device()

    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = args.gradient_accumulation_steps
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0

    args.fp32 = not ds_config["fp16"]["enabled"] 
    args.deepspeed_config = None

    distiller = Distiller(args, device)
    dataset = prepare_dataset(args, distiller)
    distiller, _, _, _ = deepspeed.initialize(
        model=distiller,
        optimizer=None,
        args=args,
        lr_scheduler=None,
        mpu=None,
        config_params=ds_config
    )
    
    if args.task == "analyse_main":
        analyse_main(args, distiller, dataset["test"], device)
    else:
        raise NotImplementedError
    
    
if __name__ == "__main__":
    main()