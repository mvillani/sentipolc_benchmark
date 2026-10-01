import json
import logging
import os
import re
import sys
import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from openai import OpenAI

# ==============================================================================
# 1. CONFIGURAZIONE UNIFICATA (MODE: "random" | "semantic")
# ==============================================================================
MODE = "semantic"  # Imposta "random" oppure "semantic"
SELECTED_MODEL = "google/gemma-3-12b-it"
EVAL_ENGINE = "OpenRouter API"
BATCH_SIZE = 64

model_slug = (
    SELECTED_MODEL.replace("/", "_").replace("-", "_").replace(".", "_")
)

if MODE == "random":
  STRATEGY_TYPE = "3-Shot Random (In-Context Learning - Seed 42 Standard)"
  INPUT_CSV = "test_set_with_random_examples_seed42.csv"
  TEST_DATASET_NAME = (
      "EVALITA SentiPOLC 2016 Gold Test (with Random Examples Seed42)"
  )
  FINAL_RESULTS_CSV = f"results_sentipolc_{model_slug}_3shot_random_seed42.csv"
  FINAL_METRICS_JSON = f"metrics_sentipolc_{model_slug}_3shot_random_seed42.json"
  LOG_FILE = f"api_response_errors_random_{model_slug}.log"
else:
  STRATEGY_TYPE = (
      "3-Shot Semantic (Dynamic Few-Shot - MiniLM-L12-v2 Embeddings)"
  )
  INPUT_CSV = (
      "test_set_with_semantic_examples_sentence_transformers_paraphrase_multilingual_MiniLM_L12_v2.csv"
  )
  TEST_DATASET_NAME = (
      "EVALITA SentiPOLC 2016 Gold Test (with Semantic Examples)"
  )
  FINAL_RESULTS_CSV = (
      f"results_sentipolc_{model_slug}_3shot_semantic_minilm.csv"
  )
  FINAL_METRICS_JSON = (
      f"metrics_sentipolc_{model_slug}_3shot_semantic_minilm.json"
  )
  LOG_FILE = f"api_response_errors_semantic_{model_slug}.log"

# ==============================================================================
# 2. SETUP LOGGING
# ==============================================================================
error_logger = logging.getLogger(f"api_errors_{MODE}")
error_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
if not error_logger.handlers:
  error_logger.addHandler(file_handler)

# ==============================================================================
# 3. SETUP API KEY OPENROUTER
# ==============================================================================
try:
  from google.colab import userdata

  openrouter_api_key = userdata.get("OPENROUTER_API_KEY")
except Exception:
  openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

if not openrouter_api_key:
  openrouter_api_key = input("🔑 Inserisci la tua OPENROUTER_API_KEY: ")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key
)

# ==============================================================================
# 4. SYSTEM PROMPT BASE & FUNZIONE PROMPT UNIFICATA
# ==============================================================================
BASE_SYSTEM_PROMPT = """Sei un annotatore esperto di Sentiment Analysis per il dataset italiano SentiPOLC.
Per ogni tweet DEVI PRIMA spiegare brevemente la motivazione del tono e POI determinare le due polarità indipendenti opos e oneg.

- opos = 1: se sono presenti parole ed espressioni di apprezzamento, supporto, gioia o elogio, altrimenti 0.
- oneg = 1: se sono presenti parole ed espressioni di critica, attacco politico, sarcasmo, ironia o insulto, altrimenti 0.
"""


def build_few_shot_prompt(row: pd.Series, mode: str) -> str:
  target_text = str(row["text"])

  if mode == "semantic":
    context = row.get("fewshot_prompt_context", "")
    return (
        f"{context}\n"
        "Ora analizza il seguente tweet target applicando i criteri appresi dagli"
        " esempi sopra:\n"
        f'Tweet da analizzare: "{target_text}"\n'
        "Fornisci la motivazione e le due polarita' opos e oneg in formato JSON:"
    )

  # Modalità "random"
  fewshot_str = "Ecco alcuni esempi guida di annotazione con la relativa motivazione e polarita':\n\n"
  for i in range(1, 4):
    ex_text = row.get(f"ex{i}_text", "")
    ex_opos = row.get(f"ex{i}_opos", 0)
    ex_oneg = row.get(f"ex{i}_oneg", 0)
    fewshot_str += (
        f'Esempio {i}:\nTweet: "{ex_text}"\nRisposta JSON:'
        f' {{"motivazione": "Esempio guida SentiPOLC", "opos": {ex_opos}, "oneg":'
        f" {ex_oneg}}}\n\n"
    )

  return (
      f"{fewshot_str}Ora analizza il seguente tweet target applicando gli stessi"
      " criteri degli esempi sopra:\n"
      f'Tweet da analizzare: "{target_text}"\n'
      "Fornisci la motivazione e le two polarita' opos e oneg in formato JSON:"
  )


# ==============================================================================
# 5. INFERENZA ENGINE UNIFICATO
# ==============================================================================
def predict_sentiment_fewshot_row(row: pd.Series) -> dict:
  user_prompt = build_few_shot_prompt(row, mode=MODE)
  tweet_id = str(row["id"])

  system_prompt = (
      f"{BASE_SYSTEM_PROMPT}\n\n"
      "Rispondi ESCLUSIVAMENTE con un oggetto JSON valido (senza testo extra o"
      " blocchi markdown) con questa struttura:\n"
      '{"motivazione": "spiegazione breve", "opos": 0 o 1, "oneg": 0 o 1}'
  )

  max_attempts = 5
  for attempt in range(1, max_attempts + 1):
    raw_text = ""
    try:
      response = client.chat.completions.create(
          model=SELECTED_MODEL,
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt},
          ],
          temperature=0.0,
          max_tokens=150,
          response_format={"type": "json_object"},
          extra_headers={
              "HTTP-Referer": "https://colab.research.google.com",
              "X-Title": f"SentiPOLC Evaluation ({MODE})",
          },
      )

      raw_text = response.choices[0].message.content or ""
      if not raw_text.strip():
        raise ValueError("Risposta API vuota")

      cleaned_text = re.sub(r"^```json\s*", "", raw_text.strip())
      cleaned_text = re.sub(r"^```\s*", "", cleaned_text)
      cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

      json_match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
      parsed_obj = (
          json.loads(json_match.group(0))
          if json_match
          else json.loads(cleaned_text)
      )

      if isinstance(parsed_obj, list) and len(parsed_obj) > 0:
        parsed_obj = parsed_obj[0]

      if isinstance(parsed_obj, dict):
        opos = (
            1
            if str(parsed_obj.get("opos", 0)).lower() in ["1", "true"]
            else 0
        )
        oneg = (
            1
            if str(parsed_obj.get("oneg", 0)).lower() in ["1", "true"]
            else 0
        )

        return {
            "motivazione": str(parsed_obj.get("motivazione", ""))[:200],
            "opos": opos,
            "oneg": oneg,
            "raw_prompt_user": user_prompt,
            "raw_system_prompt": system_prompt,
            "raw_response": raw_text,
        }

    except Exception as e:
      error_msg = (
          f"TWEET_ID: {tweet_id} | TENTATIVO: {attempt}/{max_attempts} | ERRORE:"
          f" {type(e).__name__}: {e}\n"
          f"--- USER PROMPT ---\n{user_prompt}\n"
          f"--- RAW RESPONSE ---\n'{raw_text}'\n"
          + ("-" * 80)
      )
      error_logger.error(error_msg)
      time.sleep(attempt * 2)

  return {
      "motivazione": "FALLIMENTO_API_DOPO_5_TENTATIVI",
      "opos": 0,
      "oneg": 0,
      "raw_prompt_user": user_prompt,
      "raw_system_prompt": system_prompt,
      "raw_response": "ERROR",
  }


# ==============================================================================
# 6. CARICAMENTO DATASET & LOOP SPERIMENTALE
# ==============================================================================
df_test = pd.read_csv(INPUT_CSV)
results = []
processed_ids = set()

if os.path.exists(FINAL_RESULTS_CSV):
  try:
    existing_df = pd.read_csv(FINAL_RESULTS_CSV)
    results = existing_df.to_dict("records")
    processed_ids = set(existing_df["id"].astype(str).tolist())
    print(
        f"🔄 CHECKPOINT TROVATO! Ripresa dall ID: {len(processed_ids)} /"
        f" {len(df_test)} tweet già completati.\n"
    )
  except Exception as e:
    print(f"⚠️ Errore lettura checkpoint: {e}")

start_time = time.time()
print(
    f"🚀 Avvio/Ripresa analisi [{STRATEGY_TYPE}] per [{SELECTED_MODEL}] su"
    f" {len(df_test)} tweet...\n"
)

for idx, row in df_test.iterrows():
  tweet_id = str(row["id"])
  if tweet_id in processed_ids:
    continue

  pred = predict_sentiment_fewshot_row(row)

  res_item = {
      "id": tweet_id,
      "text": str(row["text"]),
      "target_opos": int(row["opos"]),
      "target_oneg": int(row["oneg"]),
      "pred_opos": pred["opos"],
      "pred_oneg": pred["oneg"],
      "motivazione": pred.get("motivazione", ""),
      "raw_system_prompt": pred.get("raw_system_prompt", ""),
      "raw_response": pred.get("raw_response", ""),
  }

  results.append(res_item)
  print(".", end="", flush=True)

  if (len(results)) % BATCH_SIZE == 0 or (idx + 1) == len(df_test):
    current_count = len(results)
    elapsed = time.time() - start_time
    speed = current_count / elapsed if elapsed > 0 else 0

    pd.DataFrame(results).to_csv(
        FINAL_RESULTS_CSV, index=False, encoding="utf-8"
    )

    y_true_pos = [item["target_opos"] for item in results]
    y_pred_pos = [item["pred_opos"] for item in results]
    y_true_neg = [item["target_oneg"] for item in results]
    y_pred_neg = [item["pred_oneg"] for item in results]

    acc_pos = accuracy_score(y_true_pos, y_pred_pos) * 100
    acc_neg = accuracy_score(y_true_neg, y_pred_neg) * 100
    f1_pos = f1_score(y_true_pos, y_pred_pos, average="macro")
    f1_neg = f1_score(y_true_neg, y_pred_neg, average="macro")
    combined_f1 = (f1_pos + f1_neg) / 2.0

    print("\n" + "=" * 80)
    print(
        f"📦 PARZIALE {STRATEGY_TYPE} ({current_count} / {len(df_test)} tweet)"
        f" | Velocità: {speed:.2f} tweet/s"
    )
    print(f"🔹 Accuracy opos: {acc_pos:.2f}% | Accuracy oneg: {acc_neg:.2f}%")
    print(f"-> F1-Macro opos: {f1_pos:.4f} | F1-Macro oneg: {f1_neg:.4f}")
    print(f"👉 COMBINED F1-SCORE PARZIALE: {combined_f1:.4f}")
    print("=" * 80 + "\n")

# ==============================================================================
# 7. METRICHE FINALI
# ==============================================================================
if len(results) > 0:
  y_true_pos = [item["target_opos"] for item in results]
  y_pred_pos = [item["pred_opos"] for item in results]
  y_true_neg = [item["target_oneg"] for item in results]
  y_pred_neg = [item["pred_oneg"] for item in results]

  f1_pos = f1_score(y_true_pos, y_pred_pos, average="macro")
  f1_neg = f1_score(y_true_neg, y_pred_neg, average="macro")
  combined_f1 = (f1_pos + f1_neg) / 2.0

  final_metrics = {
      "model_name": SELECTED_MODEL,
      "eval_engine": EVAL_ENGINE,
      "strategy": STRATEGY_TYPE,
      "dataset": TEST_DATASET_NAME,
      "acc_opos": accuracy_score(y_true_pos, y_pred_pos) * 100,
      "acc_oneg": accuracy_score(y_true_neg, y_pred_neg) * 100,
      "f1_opos": f1_pos,
      "f1_oneg": f1_neg,
      "combined_f1": combined_f1,
      "total_tweets": len(results),
      "elapsed_time_sec": time.time() - start_time,
  }

  pd.DataFrame(results).to_csv(
      FINAL_RESULTS_CSV, index=False, encoding="utf-8"
  )
  with open(FINAL_METRICS_JSON, "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, ensure_ascii=False, indent=2)

  print("\n" + "=" * 80)
  print(f"🎉 VALUTAZIONE {STRATEGY_TYPE} COMPLETATA PER {SELECTED_MODEL}!")
  print(f"🔹 Accuracy opos: {final_metrics['acc_opos']:.2f}%")
  print(f"🔹 Accuracy oneg: {final_metrics['acc_oneg']:.2f}%")
  print(f"-> F1-Macro opos: {final_metrics['f1_opos']:.4f}")
  print(f"-> F1-Macro oneg: {final_metrics['f1_oneg']:.4f}")
  print(f"👉 COMBINED F1-SCORE DEFINITIVO: {final_metrics['combined_f1']:.4f}")
  print("=" * 80)
