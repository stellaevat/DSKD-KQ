---
license: apache-2.0
datasets:
- databricks/databricks-dolly-15k
language:
- en
metrics:
- rouge
base_model:
- openai-community/gpt2-xl
pipeline_tag: text-generation
---
# teacher-gpt2-1.5B

[paper](https://arxiv.org/abs/2306.08543) | [code](https://github.com/microsoft/LMOps/tree/main/minillm)

**teacher-gpt2-1.5B** is a gpt2-xlarge (1.5B) model supervised fine-tuned on [databricks-dolly-15k](https://huggingface.co/datasets/aisquared/databricks-dolly-15k).

It is used as the teacher model for MiniLLM series.


## Citation
```
@inproceedings{minillm,
  title={MiniLLM: Knowledge Distillation of Large Language Models},
  author={Gu, Yuxian and Dong, Li and Wei, Furu and Huang, Minlie},
  booktitle={Proceedings of ICLR},
  year={2024}
}
```