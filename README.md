# FedNAP

Training and evaluation code for **FedNAP: Nuisance-Aware Federated Preference Learning for LLM Alignment**, built on [FedBiscuit](https://github.com/HarliWu/FedBiscuit) and FederatedScope.

## Environment

Use Linux, Python 3.10 and a CUDA GPU. Install policy training and Auto-J in separate environments:

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

## Data and reproduction

The final split manifests record source versions or hashes, role counts, output hashes, and all pairwise intersections. [TLDR manifest](manifests/tldr.json) and [SHP manifest](manifests/shp.json). The staging commands below regenerate these splits.

Keep the TLDR and AlpacaEval evaluation files fixed. Match prompts after
Unicode, case and whitespace normalization, and exclude benchmark matches
when staging training inputs and candidate prompts. For each task, use 256
labeled comparisons for selector monitoring and 256 for CPF calibration,
with disjoint prompts across the panels. The preparation scripts record
split counts, prompt intersections and file hashes.

TLDR and SHP train cumulative niche selectors. After every fifth completed round the server scores each trained niche on the fixed monitoring panel. A strictly higher preference accuracy replaces that niche's elite; ties retain the earlier checkpoint. At the end of training the server exports one final elite per niche and records its round and accuracy in the archive manifest. CPF scores candidates and calibration pairs with these same elites. The DPO confidence exponent is 0.5 and the weight floor is 0.25.

### Evaluate the released TLDR summaries

The [Release](https://github.com/bobolovesinging/FedNAP/releases/tag/v1.0-paper-checkpoints) provides 6,553 generated summaries in official TLDR test order. The reported strict Auto-J win rate is **85.14%** (5,579 wins, 964 losses, 10 ties). Auto-J uses the original posts and human references from the official OpenAI test file, with the human reference first and the model summary second.

From the repository root on Linux, download the summaries and the fixed test file:

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

Run Auto-J in its own Python 3.10 environment with CUDA GPUs:

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

The evaluator checks the official test-file checksum and coverage before judging. It writes the decisions and win/loss/tie totals to `$EVAL/FedNAP-TLDR-summaries.json_autoj_eval_win.txt`. Strict win rate is wins divided by wins + losses + ties.

### 1. TLDR from official data to one staged directory

The preparation command fetches 19 worker comparison batches and three human-summary files from OpenAI and verifies all 22 SHA-256 digests. The repository supplies the official training-row indices and six FedNAP attribute labels for 9,990 eligible comparisons. Staging reconstructs their text from OpenAI. Use a fresh output directory. [OpenAI data](https://github.com/openai/summarize-from-feedback) Rows with matching Reddit IDs or matching post bodies (Unicode NFKC, case-folded, whitespace collapsed) are joined transitively. The fixed test takes priority. The script copies its bytes, keeps all 6,553 rows in order, then assigns the other source groups:

| Input | Assignment |
| --- | --- |
| Indexed OpenAI training comparisons with FedNAP attribute labels | Reserve the posts for head pretraining, validation and internal test, about 80/10/10 by group. The loader augments A/B order after assignment. |
| Official comparison validation (valid1) | Reserve two independent panels, each with 256 posts and one pair per post. Monitoring selects checkpoints. Calibration selects the CPF rule by label-stratified five-fold CV, then fits it on the panel. One row per source group ensures disjoint folds. |
| Official comparison test (valid2) and train | Reserve the selector test first, then train on remaining official training rows with their original worker IDs. Deduplicate repeated votes within worker and pair, dropping conflicting repeat votes. |
| Official human-summary validation | Exclude all comparison posts, head posts and final-test posts. Reserve about 80% of the remaining groups for policy validation and 20% as an auxiliary development holdout. |
| Official human-summary training | Exclude the reserved groups and the entire comparison pool. Reconstruct the original DPO source slice by removing exact comparison post texts and taking the last 40% of the remaining official training rows. Retain eligible source groups from that slice. Use one prompt per group for DPO candidate generation and the remaining groups for SFT. |

~~~bash
python scripts/prepare_tldr_release_data.py --data-root data/official-tldr
export TLDR_DATA=outputs/tldr/reproduction-data
export TLDR_SPLITS="$TLDR_DATA"
python scripts/stage_tldr.py --data-root data/official-tldr \
  --output-dir "$TLDR_DATA"
export TLDR_TEST="$TLDR_DATA/reddit-tldr-comparison/reddit-tldr_test.jsonl"
~~~

`manifest.json` in the staged directory records file hashes, row and group counts, and every pairwise source-group intersection. Staging fails on an intersection. These checks cover exact normalized bodies and linked Reddit IDs. These splits apply to new training runs.

For the pinned inputs, the staged selector train has 82,070 votes from 53 workers. SFT has 63,126 summaries over 63,084 groups. Candidate generation uses the 42,485 eligible source groups retained from the original slice. The head train/validation/test contain 7,770/1,227/993 annotated comparisons. All 66 pairwise group intersections are zero.

### 2. SHP from official data to one staged directory

Use the same fixed 805 AlpacaEval reference answers for staging and final evaluation. Both head profiles, the FedNAP and FedBis selector profiles, and the SHP DPO profiles default to the directory below. Run from the repository root with the policy environment active.

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

The command downloads pinned snapshots of Stanford SHP and PKU-SafeRLHF-prompt. It verifies the benchmark checksum, groups questions by NFKC, case folding and whitespace normalization, and deduplicates ordered preferred/rejected pairs per question. Training uses a 200-client domain-Dirichlet allocation (α=0.3, seed 0) and the Gemma 512-token train filter. It excludes AlpacaEval questions from every other role. The selector and attribute-head splits use disjoint questions drawn from the official training pool after excluding all Reddit candidate and AlpacaEval questions.

| Staged role | Assignment |
| --- | --- |
| Selector monitoring and CPF calibration | Two separate 256-question panels held out from the official training pool, one human comparison per question. |
| Attribute heads | Reserve 24,000/1,000/1,000 comparisons for head training, validation and testing from the official training pool, at most two pairs per question. For head training, prioritize questions with fewer eligible comparisons and break ties by SHA-256 order. Reserve each selected question in full. |
| Selector training and internal test | Hold out up to 512 questions from the official training pool for selector testing in SHA-256 order, using all comparisons from those questions. The remaining comparisons train the selector, with at least one question retained for each client. |
| Candidate generation and DPO | All distinct official validation and test questions outside AlpacaEval supply the Reddit pool. The SafeRLHF pool excludes questions already assigned to any role. Reddit-2 and Reddit-4 reuse the same Reddit pool across separate experiments. |
| Policy evaluation | Byte-for-byte copy of the fixed 805-reference file. |

The stage writes row and question counts, file hashes and all 45 pairwise question intersections across ten roles to one manifest. Any nonzero intersection aborts staging.

For the pinned data and Gemma tokenizer, head training reserves 26,971 comparisons across its question groups and uses 24,000 of them. Selector training retains 203,205 comparisons on 13,579 questions across 200 clients. The internal selector test contains 8,474 comparisons on 512 questions. Reddit generation uses 4,289 questions and SafeRLHF uses 44,557. All 45 pairwise question intersections are zero.

### 3. Train attribute heads, selectors and TLDR SFT

Train the attribute heads and TLDR SFT policy with the staged inputs.

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

TLDR and SHP sample 5 of 53 and 10 of 200 clients per round, respectively, for 200 rounds.

### 4. Generate candidates and fit CPF

TLDR reads the staged candidate file, checks the prompts against the fixed test, and verifies that the generated pairs cover every candidate group. SHP prompt preparation excludes
evaluation, monitoring, and calibration questions from both candidate pools.

CPF selects its fusion rule by five-fold cross-validation stratified by
comparison label. It fits the fold-specific parameters on the training folds,
then refits the selected rule on the full calibration panel.

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

Reddit-4 sets `--source reddit-4 --num-completions 4` and produces six
unordered pairs per prompt. SafeRLHF-2 uses `saferlhf.json` with
`--source saferlhf-2 --num-completions 2`. DPO checks the source-file hash, prompt identities and all expected completion-index pairs against the staged prompt pool.

### 5. Train DPO and evaluate

~~~bash
python federatedscope/llm/rlhf/main.py --cfg fedbiscuit_script/tldr/fednap_dpo.yaml \
  --preference-data-file outputs/tldr/preferences.json data.root "$TLDR_DATA" \
  model.load_from_local_pretrained_model_path "$TLDR_SFT" eval.split "['val']"
python federatedscope/llm/rlhf/main.py --cfg fedbiscuit_script/shp/fednap_dpo_gemma.yaml \
  --preference-data-file outputs/shp/preferences.json
~~~

DPO uses 150 rounds, 30 local updates, RMSprop at 1e-6, beta 0.1, batch size
4, and gradient accumulation 8. Set the YAML and checkpoint paths below
to those saved by the new runs.

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

Run Auto-J in a separate environment. For TLDR, Auto-J sees the human
reference first and the model summary second. Strict win rate divides
model wins by valid judgments, including ties.

```bash
conda create -n fednap-autoj python=3.10 -y
conda activate fednap-autoj
pip install -r environment/requirements_autoj.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export AUTOJ_MODEL=GAIR/autoj-13b
export AUTOJ_MAX_MODEL_LEN=8192
export AUTOJ_DISABLE_CUSTOM_ALL_REDUCE=1
export AUTOJ_ENFORCE_EAGER=1
python federatedscope/llm/eval/eval_for_tldr/auto_j_vllm.py --pairwise \
  --model-outputs-file outputs/tldr/new-eval/summarization.json --test-file "$TLDR_TEST"
```

For SHP:

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
