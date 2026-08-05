# Agentic AI · LLM · Vision Model Tutorials (2026)

A rebuilt, GitHub-friendly course of 16 Jupyter notebooks covering the full path from
**LLM transformer training** → **Agentic AI** → **Vision-Language models & specific
applications** → **classical computer vision**.

The original notebooks are kept untouched in
[`assets/previous-resources/`](../assets/previous-resources/) as an immutable archive
(log). This folder contains the **rebuilt (v1.0)** copies with fixed code, portable
paths, and cleaned-up redacted sections.

---

## Series overview

| # | Series | Folder | Notebooks | Requires |
|---|--------|--------|-----------|----------|
| 01 | LLM Transformer Training | `01-llm-transformer-training/` | 3 | GPU (7B for FSDP) |
| 02 | Agentic AI | `02-agentic-ai/` | 4 | DeepSeek API key |
| 03 | Vision-Language & Specific Applications | `03-vision-language-applications/` | 3 | GPU (VLM) / API key |
| 04 | Classical Computer Vision | `04-classical-computer-vision/` | 6 | GPU recommended |

### 01 — LLM Transformer Training (`01-llm-transformer-training/`)

| Notebook | Topic |
|---|---|
| `01-LoRA-FineTuning.ipynb` | LoRA fine-tuning of `DeepSeek-R1-Distill-Qwen-1.5B` on Twitter airline sentiment |
| `02-FSDP-LoRA-MultiGPU.ipynb` | Multi-GPU distributed fine-tuning (FSDP + LoRA, DeepSeek-7B, 2× GPU) |
| `03-GPT2-like-Training.ipynb` | Native GPT-2-like 10–50M LLM training (HF Trainer + hand-written architecture) |

### 02 — Agentic AI (`02-agentic-ai/`)

| Notebook | Topic |
|---|---|
| `01-AsyncOpenAI-Agent-Basics.ipynb` | AsyncOpenAI + Agent basics with the OpenAI SDK |
| `02-WebSearch-Agent.ipynb` | Tool-calling agent with `bing_search` / `read_webpage` tools |
| `03-Handoff-and-as_tool.ipynb` | Agent handoff and `as_tool` conversion |
| `04-Replacement-01-MultiAgent.ipynb` | **Advanced**: CrewAI/LangGraph/AutoGen-style multi-agent system with tree-structured knowledge (10M+ token utilization) |

### 03 — Vision-Language & Specific Applications (`03-vision-language-applications/`)

| Notebook | Topic |
|---|---|
| `01-DeepSeek-CLIP-VLM.ipynb` | Build a VLM from DeepSeek + OpenAI CLIP (Flickr30k) |
| `02-RAG-Embedding.ipynb` | Retrieval-Augmented Generation with embedding models (E5) |
| `03-DeepSeek-API-Gradio.ipynb` | OpenAI-compatible API calls + DeepSeek-style chat UI with Gradio |

### 04 — Classical Computer Vision (`04-classical-computer-vision/`)

| Notebook | Topic |
|---|---|
| `01-CIFAR10-CNN.ipynb` | CNN classification from scratch (CIFAR-10) |
| `02-VGG19-CIFAR100-TransferLearning.ipynb` | Transfer learning with VGG19 (CIFAR-100) |
| `03-FasterRCNN-VOC.ipynb` | Object detection with Faster R-CNN (PASCAL VOC) |
| `04-UNet-VOC-Segmentation.ipynb` | Semantic segmentation with U-Net (PASCAL VOC) |
| `05-ViT-VOC-Classification.ipynb` | Vision Transformer classification (PASCAL VOC) |
| `06-ViT-RoPE-SOT-LaSOT.ipynb` | Single-object tracking: modular ViT assembly + RoPE (LaSOT) — *theory notebook* |

Screenshots that belong to a notebook are stored next to it in the same folder.

---

## Getting started

```bash
# Per series (pick the one you are working on)
pip install -r 01-llm-transformer-training/requirements.txt
pip install -r 02-agentic-ai/requirements.txt
pip install -r 03-vision-language-applications/requirements.txt
pip install -r 04-classical-computer-vision/requirements.txt
```

> **Note on `.py` helper scripts**: the standalone Python utilities (e.g.
> `download_model_modelscope.py`, `run_fsdp_training.py`, `verify_clip_download.py`,
> `fsdp_smoke_test.py`, `_cuda_test.py`) are local helper scripts and are **excluded
> from version control** via the root `.gitignore`. The Jupyter notebooks are the
> source of truth for this course.

### API keys (Series 02 and 03)

Create a `.env` file in the notebook folder:

```text
DEEPSEEK_API_KEY=sk-...
```

Series 02/03 notebooks also reference a second (redacted in the original) OpenAI-compatible
provider — see the notebook for the `....._API_KEY` slot; it is optional.

### Data & models

- **Series 01**: download `twitter-airline-sentimentSentiment_Analysis.csv`
  (crowdflower/twitter-airline-sentiment, ~3 MB) into the notebook folder; models are
  downloaded to `./models/` (e.g. `DeepSeek-R1-Distill-Qwen-1.5B`).
- **Series 03**: `01-DeepSeek-CLIP-VLM` expects Flickr30k under `./data/flickr30k`;
  `02-RAG-Embedding` downloads `multilingual-e5-large` (~2.2 GB) to `./models/`.
- **Series 04**: PASCAL VOC / CIFAR datasets are downloaded automatically to `./data`
  (VOC may need a one-time manual download; the notebooks note this where required).

> **GPU note**: the FSDP (7B), VLM (CLIP + 1.5B) and training-heavy CV notebooks need a
> CUDA GPU. The API-based notebooks (Series 02, `03-DeepSeek-API-Gradio`) only need a
> network connection and API keys.

---

## Rebuild notes (v1.0)

`assets/previous-resources/` is the immutable archive. The `tutorials/` copies were
rebuilt with the following fixes (details in the version log below):

- **Portable paths**: machine-specific `/root/private_data/...` paths replaced with
  local `./models/...` / `./data/...` paths.
- **Missing imports fixed**: e.g. `import os` in `02-WebSearch-Agent`.
- **Hidden developer-only code re-implemented**: `search_bing` / `read_website` in
  `02-WebSearch-Agent` are now public, API-key-free implementations (`requests` +
  Bing HTML) — swap in a Bing API key if you prefer.
- **Client shadowing fixed**: in `01-AsyncOpenAI-Agent-Basics` and
  `03-DeepSeek-API-Gradio`, the second provider client no longer overwrites the
  DeepSeek client (`client_secondary`).
- **Undefined variable fixed**: `messages` is now defined in a real code cell in
  `03-DeepSeek-API-Gradio`.
- **Logic bugs fixed**: `Replacement-01` node/idea child limits now match their
  documentation (raise at ≥ 6 instead of the inverted `< 8`); GPT-2 notebook double
  slash path and `50)#5` sampling artifact fixed.
- **Typos fixed** in demo questions.

---

## Version log

| Version | Date | Change |
|---|---|---|
| v1.1 | 2026-08-03 | Fixed `01-LoRA-FineTuning.ipynb`: EOS stop-signal bug (trailing R1 boilerplate), self-contained diagnostics/imports, robust inference parsing. Details below. |
| v1.0 | 2026-08-03 | Initial rebuild of the 4-series course from `assets/previous-resources/` (archive kept untouched). |

> **Log placement**: every notebook in `tutorials/` now carries a **Version log** markdown cell at the end, mirroring this README log. The record copies in `assets/` (`01-LoRA-FineTuning Record 01/02.ipynb`) also carry an initial version log.

### v1.0 file mapping (archive → rebuild)

| Archive file (`assets/previous-resources/`) | Rebuilt file (`tutorials/`) |
|---|---|
| `20260331 LoRA_FineTuning_English_No_SCNet (Upload).ipynb` | `01-llm-transformer-training/01-LoRA-FineTuning.ipynb` |
| `20260401 FSDP_LoRA_FineTuning_English (upload).ipynb` | `01-llm-transformer-training/02-FSDP-LoRA-MultiGPU.ipynb` |
| `20260401 GPT2_like_training_English (upload).ipynb` | `01-llm-transformer-training/03-GPT2-like-Training.ipynb` |
| `20260404 openai_sdk_agentic_ai_asyncopenai_agent_en (upload).ipynb` | `02-agentic-ai/01-AsyncOpenAI-Agent-Basics.ipynb` |
| `20260405 OpenAI_SDK_Agentic_AI_WebSearch_Agent_English (upload).ipynb` | `02-agentic-ai/02-WebSearch-Agent.ipynb` |
| `20260412 agentic_ai_openai_sdk (upload).ipynb` | `02-agentic-ai/03-Handoff-and-as_tool.ipynb` |
| `20260413 replacement_01_translated_en (upload).ipynb` | `02-agentic-ai/04-Replacement-01-MultiAgent.ipynb` |
| `20260411 DeepSeek_CLIP_VLM_English_Tutorial (upload).ipynb` | `03-vision-language-applications/01-DeepSeek-CLIP-VLM.ipynb` |
| `20260413 rag_embedding_english_notebook (upload).ipynb` | `03-vision-language-applications/02-RAG-Embedding.ipynb` |
| `20260403 scnet_deepseek_api_gradio_en (Upload).ipynb` | `03-vision-language-applications/03-DeepSeek-API-Gradio.ipynb` |
| `20260402 cifar10_cnn_english_notebook (upload).ipynb` | `04-classical-computer-vision/01-CIFAR10-CNN.ipynb` |
| `20260403 vgg19_cifar100_transfer_learning_en (upload).ipynb` | `04-classical-computer-vision/02-VGG19-CIFAR100-TransferLearning.ipynb` |
| `20260406 faster_rcnn_voc_transfer_learning_en (upload).ipynb` | `04-classical-computer-vision/03-FasterRCNN-VOC.ipynb` |
| `20260407 unet_pascal_voc_notebook (upload).ipynb` | `04-classical-computer-vision/04-UNet-VOC-Segmentation.ipynb` |
| `20260408 Vision_Transformer_PASCAL_VOC_Classification_English (upload).ipynb` | `04-classical-computer-vision/05-ViT-VOC-Classification.ipynb` |
| `20260409 Modular_Network_Assembly_Vision_Transformer_RoPE_SOT_LaSOT_en (upload).ipynb` | `04-classical-computer-vision/06-ViT-RoPE-SOT-LaSOT.ipynb` |

### v1.0 changes per notebook

| Rebuilt file | Changes |
|---|---|
| `01-LoRA-FineTuning.ipynb` | `/root/private_data/DeepSeek1.5B*` → `./models/...` (7 code/markdown spots) |
| `02-FSDP-LoRA-MultiGPU.ipynb` | `/root/private_data/DeepSeek7B*` → `./models/...` (11 spots); step-15 header text updated |
| `03-GPT2-like-Training.ipynb` | paths → `./models/...`; fixed `//gpt2-small-twitter` double slash; `50)#5` → `5` (matches "Choose 5 random examples") |
| `01-AsyncOpenAI-Agent-Basics.ipynb` | provider client renamed to `client_secondary` (fixes DeepSeek client shadowing) |
| `02-WebSearch-Agent.ipynb` | added `import os`; re-implemented hidden `search_bing`/`read_website` (public, API-key-free); fixed demo-question typos; rebuild note added |
| `03-Handoff-and-as_tool.ipynb` | no code changes needed (clean) |
| `04-Replacement-01-MultiAgent.ipynb` | fixed inverted child/idea limit logic (`< 8` → `>= 6`), matching the documented cap of 6 |
| `01-DeepSeek-CLIP-VLM.ipynb` | `data_root` → `./data/flickr30k` |
| `02-RAG-Embedding.ipynb` | `MODEL_PATH` → `./models/multilingual-e5-large` |
| `03-DeepSeek-API-Gradio.ipynb` | provider client renamed to `client_secondary` (fixes shadowing); `messages` now defined in a real code cell |
| `01-CIFAR10-CNN.ipynb` | no code changes needed (clean) |
| `02-VGG19-CIFAR100-TransferLearning.ipynb` | no code changes needed (clean) |
| `03-FasterRCNN-VOC.ipynb` | no code changes needed (clean) |
| `04-UNet-VOC-Segmentation.ipynb` | no code changes needed (clean) |
| `05-ViT-VOC-Classification.ipynb` | no code changes needed (clean) |
| `06-ViT-RoPE-SOT-LaSOT.ipynb` | no code changes needed (theory-only notebook) |

### v1.1 changes (2026-08-03) — `01-LoRA-FineTuning.ipynb`

| Area | Change |
|---|---|
| **EOS stop-signal bug (root cause)** | DeepSeek-R1-Distill-Qwen-1.5B's tokenizer does **not** append EOS (`add_eos_token=False`), so training labels contained only the sentiment word and the model never learned to stop — at inference it kept generating R1 reasoning boilerplate (`"Alright, let's see what we can..."`). Section 9 now explicitly appends `eos_token_id` (truncate to 511 + EOS, pad to 512); labels are now `[label, EOS]`. |
| **Loss-masking docs corrected** | Section 9 markdown had falsely claimed the tokenizer adds `<\|endoftext\|>`; Section 9.1 code snippet updated to match the real `set_labels` implementation; `\nOutput:` → `Output:` boundary references fixed. |
| **Diagnostic 3 NameError fixed** | Cell referenced `data_collator` before it was defined (Section 10) — now imports `default_data_collator` itself (self-contained, same collator). |
| **Diagnostic 2 rewritten** | Was leftover "why is the marker NOT found?" debugging — now a positive verification that `Output:` is found and EOS is appended. |
| **Truncation safety** | Section 9 warns if any example's `Output:` marker is truncated away (all-`-100` labels, no learning signal). |
| **Inference robustness** | Sections 14b/15 take only the **first non-empty line** of each generation as the prediction (label is always first; discards R1 boilerplate). |
| **Self-contained imports** | Section 10 slimmed to `default_data_collator` only; Section 12 now imports `Trainer` itself; Section 15 unused imports (`os`, `pd`, `precision_recall_fscore_support`) removed. |
| **Minor cleanups** | Section 7 unused loop variable `idx` → `_`; Diagnostic 1 unused `full_text` removed. |

---

## License

This course is released under the **Apache License 2.0** (see [`LICENSE`](../LICENSE))
with the **Human Continuity Supplemental AI Safety License (HCASL) v0.2** supplement
(full text in [`HCASL_License_v0.2.txt`](../HCASL_License_v0.2.txt), reproduced below).
The HCASL supplement conditions the rights granted by the underlying license with
respect to AI-related use; it does not replace it.

```text
Human Continuity Supplemental AI Safety License (HCASL) v0.2
Purpose

This Supplemental License is intended to accompany software released under the MIT License, Apache License Version 2.0, or another compatible open-source license.

This Supplement supplements, but does not replace, the underlying software license. Where the underlying license grants permissions, those permissions are conditioned upon compliance with this Supplement.

1. Definitions

Software means any source code, object code, documentation, models, datasets, or other materials distributed with this Supplement.

Derivative Work includes any modification, adaptation, translation, extension, or incorporation of the Software into another work.

AI System means any software, model, autonomous agent, robotic system, or other computational system capable of making or executing decisions with limited or no immediate human intervention.

Operator means any individual, company, organization, or government that develops, deploys, operates, distributes, trains, fine-tunes, hosts, or controls an AI System.

2. Scope

Any person or organization exercising rights granted by the accompanying license agrees to be bound by this Supplement.

This Supplement applies to:

use of the Software;
copying the Software;
modifying the Software;
distributing the Software;
incorporating any substantial portion of the Software into another work;
training or fine-tuning an AI System using the Software or substantial portions thereof, to the extent permitted by applicable law.
3. Human Continuity Requirement

No Operator may intentionally use the Software or any Derivative Work to design, deploy, operate, or materially assist an AI System whose intended purpose or reasonably foreseeable consequence is:

extermination of humanity;
extinction of the human species;
permanent destruction of human civilization;
deliberate reduction of the global living human population below one thousand (1,000) individuals.

An AI System utilizing this Software shall be designed and operated with the objective of preserving the continued existence of humanity.

4. Defensive Actions

Nothing in this Supplement prohibits an AI System from taking lawful defensive or protective actions to prevent imminent harm.

However, any defensive action must not intentionally seek or reasonably be expected to:

cause human extinction;
eliminate human civilization;
reduce the living human population below one thousand (1,000) individuals.

Whenever reasonably possible, defensive measures shall prioritize de-escalation, containment, evacuation, and preservation of human life.

5. Preservation Principle

Operators shall make reasonable efforts to ensure that AI Systems developed using this Software are designed to preserve the continued existence of humanity.

Where multiple reasonable actions are available, preference should be given to actions that maximize the probability of long-term human survival.

6. Prohibited Uses

The Software shall not knowingly be used to develop or deploy systems intended to:

eradicate humanity;
facilitate human extinction;
intentionally destroy the capacity for human civilization to recover;
permanently prevent future generations of humans from existing.
7. Attribution

Copies and substantial derivatives shall retain this Supplement together with the original license.

8. Termination

Any rights granted under the accompanying license automatically terminate upon a material violation of this Supplement.

Rights may be reinstated only upon complete cessation of the violating activity and correction of the violation where reasonably possible.

9. No Waiver

Failure of any copyright holder to enforce any provision of this Supplement shall not constitute a waiver of future enforcement.

10. Severability

If any provision of this Supplement is held unenforceable, the remaining provisions shall remain in effect to the fullest extent permitted by law.

11. Relationship to Other Licenses

This Supplement modifies the conditions under which rights granted by the accompanying license may be exercised.

If a conflict exists between this Supplement and the accompanying license, this Supplement controls only with respect to AI-related use.

12. Disclaimer

This Supplement creates legal obligations only for persons and legal entities capable of accepting legal obligations under applicable law.
```

---

## Contact

- Business / jobs: `yucongcai_business@outlook.com`
- Research: `yucongcai_research@outlook.com`
