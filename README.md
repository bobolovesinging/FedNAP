# FedNAP

Training and evaluation code for **FedNAP: Nuisance-Aware Federated Preference Learning for LLM Alignment**, built on [FedBiscuit](https://github.com/HarliWu/FedBiscuit) and FederatedScope.

[Re-evaluate the released summaries](#evaluate-the-released-tldr-summaries) or [reproduce the training workflow](#data-and-reproduction).

## Evaluate the released TLDR summaries

The [Release](https://github.com/bobolovesinging/FedNAP/releases/tag/v1.0-paper-checkpoints) contains **6,553 generated summaries** in official TLDR test order. The reported strict Auto-J win rate is **85.14%** (5,579 wins, 964 losses, 10 ties). Auto-J compares each model summary against the human reference, with the reference first. Re-evaluation uses the released summaries and the fixed OpenAI test file.

From the repository root on Linux, download the two inputs and verify the summary checksum.

~~~bash
set -euo pipefail
EVAL=outputs/tldr/public-eval
mkdir -p "$EVAL"
curl -fL --retry 3 -o "$EVAL/FedNAP-TLDR-summaries.json" \
  https://github.com/bobolovesinging/FedNAP/releases/download/v1.0-paper-checkpoints/FedNAP-TLDR-summaries.json
echo "77bc3c940ad46bf4efb405646276afdcbb618a2a2df89756ad3eef2cecabb3d5  $EVAL/FedNAP-TLDR-summaries.json" | sha256sum -c -
curl -fL --retry 3 -o "$EVAL/test.jsonl" \
  https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/tldr_3_filtered/test.jsonl
~~~

Run Auto-J in a dedicated Python 3.10 environment with a CUDA GPU.

~~~bash
conda create -n fednap-autoj python=3.10 -y
conda activate fednap-autoj
pip install -r environment/requirements_autoj.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export AUTOJ_MODEL=GAIR/autoj-13b AUTOJ_MAX_MODEL_LEN=8192
python federatedscope/llm/eval/eval_for_tldr/auto_j_vllm.py --pairwise \
  --model-outputs-file "$EVAL/FedNAP-TLDR-summaries.json" --test-file "$EVAL/test.jsonl"
~~~

The evaluator verifies the test checksum and coverage. It writes decisions and win/loss/tie totals to `$EVAL/FedNAP-TLDR-summaries.json_autoj_eval_win.txt`. Strict win rate divides wins by all valid judgments, including ties.

## Data and reproduction

The [TLDR manifest](manifests/tldr.json) and [SHP manifest](manifests/shp.json) record the staged counts, source versions, file hashes, and intersections between data roles. Staging preserves the fixed benchmarks, excludes their source groups from other roles, and fails on cross-role overlap. Each task uses separate 256-pair panels for selector monitoring and CPF calibration.

| Task | Fixed evaluation | Selector training | DPO candidate prompts |
| --- | ---: | ---: | --- |
| TLDR | 6,553 | 82,070 comparisons across 53 clients | 42,485 OpenAI TLDR prompts |
| SHP | 805 | 203,205 comparisons across 200 clients | 4,289 Reddit questions and 44,557 SafeRLHF prompts |

### Environment

Use Linux, Python 3.10, and a CUDA GPU. Install the training environment separately from Auto-J.

```bash
conda create -n fednap python=3.10 -y
conda activate fednap
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r environment/requirements_fednap.txt
pip install -e . --no-deps
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

Download base models under their upstream terms. Qwen selectors use `Qwen/Qwen2-0.5B`. The TLDR policy uses `AI-ModelScope/gemma-2b`; SHP policies use `mlabonne/Gemmalpaca-2B` or `NEU-HAI/Llama-2-7b-alpaca-cleaned`. Save snapshots under the paths used in each YAML profile, or update `model.type`.

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-0.5B', local_dir='models/Qwen2-0.5B')"
# In an environment with modelscope installed, after accepting Gemma's terms:
python -c "from modelscope import snapshot_download; snapshot_download('AI-ModelScope/gemma-2b', local_dir='models/gemma-2b')"
```

### 1. TLDR from official data to one staged directory

The preparation script downloads and checksums the [official OpenAI data](https://github.com/openai/summarize-from-feedback). Staging groups rows by Reddit ID or post body normalized with Unicode NFKC, case folding, and whitespace collapse. It keeps the official test file byte for byte.

~~~bash
python scripts/prepare_tldr_release_data.py --data-root data/official-tldr
export TLDR_DATA=outputs/tldr/reproduction-data
export TLDR_SPLITS="$TLDR_DATA"
python scripts/stage_tldr.py --data-root data/official-tldr \
  --output-dir "$TLDR_DATA"
export TLDR_TEST="$TLDR_DATA/reddit-tldr-comparison/reddit-tldr_test.jsonl"
~~~

| Source | Staged roles |
| --- | --- |
| Official comparison train | Attribute-head train/validation/test from indexed labels (7,770/1,227/993 pairs) and selector training by worker ID from the remaining eligible comparisons |
| Official comparison validation and test | Separate monitoring and calibration panels from `valid1` and selector test from `valid2` (50,567 comparisons) |
| Official human-summary validation | Policy validation and development holdout (457 and 115 prompts) |
| Official human-summary train | SFT (63,126 summaries) and DPO candidate generation (42,485 prompts) |

To construct the candidate pool, staging removes exact comparison post texts from the official human-summary training rows and takes the last 40% in official order. It retains one eligible prompt per source group. The remaining eligible training groups supply SFT.

### 2. SHP from official data to one staged directory

Stage pinned Stanford SHP and PKU-SafeRLHF-prompt data using the training environment and Gemma tokenizer. The script verifies the fixed AlpacaEval reference file. It groups questions by normalized text and assigns eligible training data to 200 clients by domain using Dirichlet $\alpha=0.3$ (seed 0) after the 512-token comparison filter.

~~~bash
export SHP_SPLITS=outputs/shp/reproduction-data
export SHP_DATA="$SHP_SPLITS/selector"
export SHP_REFERENCE="$SHP_SPLITS/alpacaeval.json"
mkdir -p data/shp
curl -fL --retry 3 -o data/shp/alpacaeval.json \
  https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/34e01fac2c393593366f2f62abd88507e00d6399/results/gpt4_1106_preview/model_outputs.json
python -m scripts.stage_shp --tokenizer-path models/gemma-2b \
  --evaluation data/shp/alpacaeval.json --output-dir "$SHP_SPLITS"
~~~

| Source | Staged roles |
| --- | --- |
| Official SHP train | Attribute-head train/validation/test (24,000/1,000/1,000 pairs, at most two per question), selector training, selector test (8,474 comparisons on 512 questions), monitoring, and calibration |
| Official SHP validation and test | Reddit candidate prompts for separate Reddit-2 and Reddit-4 runs |
| SafeRLHF | Candidate prompts from questions excluded from other roles |
| Fixed AlpacaEval references | 805 policy-evaluation instructions |

### 3. Train attribute heads, selectors and TLDR SFT

Train on the staged inputs. TLDR samples 5 of 53 clients per round and SHP samples 10 of 200, each over 200 rounds. Every fifth completed round, the server evaluates each trained niche on the monitoring panel. It exports the highest-accuracy checkpoint per niche at the end of training. Ties retain the earlier checkpoint.

~~~bash
python federatedscope/main.py --cfg fedbiscuit_script/tldr/fednap_heads.yaml \
  data.labeled_json_path "$TLDR_SPLITS/training/head_annotations.json"
export TLDR_HEAD=outputs/tldr/head-pretraining/adv_pretrained.pt
python federatedscope/main.py --cfg fedbiscuit_script/tldr/fednap_periodic.yaml \
  data.root "$TLDR_DATA" data.worker_data_dir "$TLDR_DATA/tldr_full" eval.split "['val']" \
  trainer.adv_pretrained_heads.bias_length "$TLDR_HEAD" \
  trainer.adv_pretrained_heads.bias_lead "$TLDR_HEAD"
python federatedscope/main.py --cfg fedbiscuit_script/tldr/fednap_sft.yaml \
  data.root "$TLDR_DATA" eval.split "['val']"
export TLDR_SFT=outputs/tldr/sft/final_policy.ckpt

export FEDBISCUIT_FORCE_BF16=1
python federatedscope/main.py --cfg fedbiscuit_script/shp/fednap_head_length.yaml
python federatedscope/main.py --cfg fedbiscuit_script/shp/fednap_head_person.yaml
export SHP_LENGTH_HEAD=/path/to/new/length/adv_pretrained.pt
export SHP_PERSON_HEAD=/path/to/new/person/adv_pretrained.pt
python federatedscope/main.py --cfg fedbiscuit_script/shp/fednap_periodic.yaml \
  data.root "$SHP_DATA" data.prepared_dir "$SHP_DATA" \
  trainer.adv_pretrained_heads.bias_length "$SHP_LENGTH_HEAD" \
  trainer.adv_pretrained_heads.bias_first_person "$SHP_PERSON_HEAD"
~~~

### 4. Generate candidates and fit CPF

Score candidates and calibration pairs with the exported elite selectors. CPF selects its fusion rule with five-fold cross-validation stratified by comparison label, then fits the selected rule on the full calibration panel. The confidence exponent is 0.5 and the weight floor is 0.25.

~~~bash
python scripts/generate_tldr_candidates.py \
  --policy-config fedbiscuit_script/tldr/fednap_sft.yaml \
  --checkpoint "$TLDR_SFT" --data-root "$TLDR_DATA" --evaluation "$TLDR_TEST" \
  --output outputs/tldr/pairs.json
python scripts/cache_map_elites_margins.py \
  --selector-config fedbiscuit_script/tldr/fednap_periodic.yaml --task tldr \
  --pair-data outputs/tldr/pairs.json --validation-data "$TLDR_SPLITS/calibration.jsonl" \
  --cell-dir outputs/tldr/selector/selector_final_cells --output-dir outputs/tldr/margins \
  --validation-size 256
python scripts/build_shp_confidence_preferences.py \
  --pair-data outputs/tldr/pairs.json --margin-manifest outputs/tldr/margins/margin_manifest.json \
  --output outputs/tldr/preferences.json --summary-output outputs/tldr/fusion.json --gamma 0.5

python scripts/generate_shp_candidates.py \
  --prompt-data "$SHP_SPLITS/prompts/reddit.json" \
  --policy-config fedbiscuit_script/shp/fednap_dpo_gemma.yaml \
  --policy gemma --policy-revision YOUR_BASE_MODEL_REVISION --source reddit-2 \
  --num-completions 2 --output outputs/shp/pairs.json --manifest-output outputs/shp/candidates.json
python scripts/cache_map_elites_margins.py \
  --selector-config fedbiscuit_script/shp/fednap_periodic.yaml --task shp \
  --pair-data outputs/shp/pairs.json --validation-data "$SHP_SPLITS/calibration.jsonl" \
  --cell-dir outputs/shp/selector/selector_final_cells --output-dir outputs/shp/margins \
  --validation-size 256
python scripts/build_shp_confidence_preferences.py \
  --pair-data outputs/shp/pairs.json --margin-manifest outputs/shp/margins/margin_manifest.json \
  --output outputs/shp/preferences.json --summary-output outputs/shp/fusion.json --gamma 0.5
~~~

For Reddit-4, set `--source reddit-4 --num-completions 4` to form six unordered pairs per prompt. For SafeRLHF-2, pass `saferlhf.json` with `--source saferlhf-2 --num-completions 2`. The scripts check the staged prompt identities and candidate coverage.

### 5. Train DPO and evaluate

~~~bash
python federatedscope/llm/rlhf/main.py --cfg fedbiscuit_script/tldr/fednap_dpo.yaml \
  --preference-data-file outputs/tldr/preferences.json data.root "$TLDR_DATA" \
  model.load_from_local_pretrained_model_path "$TLDR_SFT" eval.split "['val']"
python federatedscope/llm/rlhf/main.py --cfg fedbiscuit_script/shp/fednap_dpo_gemma.yaml \
  --preference-data-file outputs/shp/preferences.json
~~~

DPO uses 150 rounds, 30 local updates, RMSprop at 1e-6, beta 0.1, batch size 4, and gradient accumulation 8. Set the YAML and checkpoint paths below to the files saved by your runs.

~~~bash
python scripts/eval_tldr_policy_direct_leftpad.py \
  --cfg "$TLDR_DPO_CONFIG" --ckpt "$TLDR_DPO_CHECKPOINT" \
  --test-file "$TLDR_TEST" \
  --expected-test-sha256 82c859ca9b1ded55cff8679952feafdd2f502c8a525dc0703339de6069188bb8 \
  --outdir outputs/tldr/new-eval

export ALPACA_EVAL_REFERENCE_FILE="$SHP_REFERENCE"
export ALPACA_EVAL_OUTPUT_PATH=outputs/shp/new-eval/alpaca_eval.json
python federatedscope/llm/eval/eval_for_AlpacaEval/eval_winrate.py \
  --cfg "$SHP_DPO_CONFIG" federate.save_to "$SHP_DPO_CHECKPOINT"
~~~

Use the Auto-J environment from the released-summary instructions. TLDR uses the human reference first.

```bash
conda activate fednap-autoj
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export AUTOJ_MODEL=GAIR/autoj-13b
export AUTOJ_MAX_MODEL_LEN=8192
export AUTOJ_DISABLE_CUSTOM_ALL_REDUCE=1
export AUTOJ_ENFORCE_EAGER=1
python federatedscope/llm/eval/eval_for_tldr/auto_j_vllm.py --pairwise \
  --model-outputs-file outputs/tldr/new-eval/summarization.json --test-file "$TLDR_TEST"
```

For SHP, use the same Auto-J environment.

~~~bash
AUTOJ_MAX_MODEL_LEN=4096 python scripts/eval_shp_label_fixed_autoj.py \
  --generation-file outputs/shp/new-eval/alpaca_eval.json \
  --output-dir outputs/shp/new-autoj --policy gemma --method map_elites \
  --reference "$SHP_REFERENCE"
~~~

## Implementation map

| Component | Source |
| --- | --- |
| Data rules and descriptors | `federatedscope/llm/dataset/tldr_preference.py`, `shp_preference.py` |
| Frozen-head regularization | `federatedscope/llm/trainer/fedcqd_trainer.py`, `reward_choice_causal_trainer.py` |
| Niche aggregation and archive | `federatedscope/llm/workers/map_elites_server.py` |
| Checkpoint loading and voting | `federatedscope/llm/map_elites_voting.py` |
| Temperature fitting and fusion | `federatedscope/llm/map_elites_improvement.py` |
| Confidence-weighted DPO | `federatedscope/llm/trainer/reward_trainer.py`, `rlhf/preference_data.py` |

## Licenses

The code uses the Apache-2.0 license. See [LICENSE](LICENSE) for upstream notices. Base models and datasets retain their upstream terms.
