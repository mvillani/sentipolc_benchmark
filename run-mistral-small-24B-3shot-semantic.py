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
# 1. CONFIGURAZIONE TRI-MODALE (MODE: "0shot" | "random" | "semantic")
# ==============================================================================
MODE = "semantic"  # <-- Imposta "random" oppure "semantic" (o "0shot")
SELECTED_MODEL = "mistralai/mistral-small-24b-instruct-2501"
EVAL_ENGINE = "OpenRouter API"
BATCH_SIZE = 64

model_slug = (
    SELECTED_MODEL.replace("/", "_").replace("-", "_").replace(".", "_")
)

# Gestione automatica dei file di input e dei percorsi di output
if MODE == "0shot":
  STRATEGY_TYPE = "0-Shot Pure (Zero-Shot Direct JSON Prompting)"
  INPUT_CSV = "test_set_sentipolc16_gold2000.csv"
  TEST_DATASET_NAME = "EVALITA SentiPOLC 2016 Gold Test"
  FINAL_RESULTS_CSV = f"results_sentipolc_{model_slug}_zeroshot.csv"
  FINAL_METRICS_JSON = f"metrics_sentipolc_{model_slug}_zeroshot.json"
  LOG_FILE = f"api_response_errors_zeroshot_{model_slug}.log"

elif MODE == "random":
  STRATEGY_TYPE = "3-Shot Random (In-Context Learning - Seed 42 Standard)"
  INPUT_CSV = "test_set_with_random_examples_seed42.csv"
  TEST_DATASET_NAME = (
      "EVALITA SentiPOLC 2016 Gold Test (with Random Examples Seed42)"
  )
  FINAL_RESULTS_CSV = f"results_sentipolc_{model_slug}_3shot_random_seed42.csv"
  FINAL_METRICS_JSON = f"metrics_sentipolc_{model_slug}_3shot_random_seed42.json"
  LOG_FILE = f"api_response_errors_random_{model_slug}.log"

elif MODE == "semantic":
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
else:
  raise ValueError("MODE non valido. Scegli tra '0shot', 'random', 'semantic'.")

# ==============================================================================
# 2. SETUP LOGGING DEDICATO PER ERRORI API
# ==============================================================================
error_logger = logging.getLogger(f"api_errors_{MODE}_{model_slug}")
error_logger.setLevel(logging.INFO)
error_logger.handlers.clear()
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
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
# 4. SYSTEM PROMPT BASE & COSTRUTTORE UNIFICATO PROMPT FEW-SHOT
# ==============================================================================
BASE_SYSTEM_PROMPT = """Sei un annotatore esperto di Sentiment Analysis per il dataset italiano SentiPOLC.
Per ogni tweet DEVI PRIMA spiegare brevemente la motivazione del tono e POI determinare le due polarità indipendenti opos e oneg.

- opos = 1: se sono presenti parole ed espressioni di apprezzamento, supporto, gioia o elogio, altrimenti 0.
- oneg = 1: se sono presenti parole ed espressioni di critica, attacco politico, sarcasmo, ironia o insulto, altrimenti 0.
"""


def build_unified_prompt(row: pd.Series, mode: str) -> str:
  target_text = str(row["text"])

  if mode == "0shot":
    return (
        f'Tweet da analizzare: "{target_text}"\nFornisci la motivazione e le due'
        " polarita' opos e oneg in formato JSON:"
    )

  elif mode == "semantic":
    context = row.get("fewshot_prompt_context", "")
    return (
        f"{context}\n"
        "Ora analizza il seguente tweet target applicando i criteri appresi dagli"
        " esempi sopra:\n"
        f'Tweet da analizzare: "{target_text}"\n'
        "Fornisci la motivazione e le due polarita' opos e oneg in formato JSON:"
    )

  elif mode == "random":
    fewshot_str = "Ecco alcuni esempi guida di annotazione con la relativa motivazione e polarita':\n\n"
    for i in range(1, 4):
      ex_text = row.get(f"ex{i}_text", "")
      ex_opos = row.get(f"ex{i}_opos", 0)
      ex_oneg = row.get(f"ex{i}_oneg", 0)
      fewshot_str += (
          f'Esempio {i}:\nTweet: "{ex_text}"\nRisposta JSON:'
          f' {{"motivazione": "Esempio guida SentiPOLC", "opos": {ex_opos},'
          f' "oneg": {ex_oneg}}}\n\n'
      )

    return (
        f"{fewshot_str}Ora analizza il seguente tweet target applicando gli"
        " stessi criteri degli esempi sopra:\n"
        f'Tweet da analizzare: "{target_text}"\n'
        "Fornisci la motivazione e le due polarita' opos e oneg in formato JSON:"
    )


# ==============================================================================
# 5. INFERENZA ENGINE DEDICATO PER MISTRAL (CON PARSING JSON E FALLBACK)
# ==============================================================================
TRUE_VALUES = {"1", "true", "yes", "y", "t", "1.0"}


def parse_json_response(raw_text: str) -> dict:
  """Estrattore JSON robusto universale per Mistral Small."""
  cleaned = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.IGNORECASE)
  cleaned = re.sub(r"^```\s*", "", cleaned)
  cleaned = re.sub(r"\s*```$", "", cleaned)

  json_match = re.search(r"\{[\s\S]*?\}", cleaned)
  json_str = json_match.group(0) if json_match else cleaned

  parsed_obj = json.loads(json_str)
  if isinstance(parsed_obj, list) and len(parsed_obj) > 0:
    parsed_obj = parsed_obj[0]

  if isinstance(parsed_obj, dict):
    # Gestione eventuali strutture annidate
    if not ("opos" in parsed_obj or "oneg" in parsed_obj):
      for k, v in parsed_obj.items():
        if isinstance(v, dict) and ("opos" in v or "oneg" in v):
          parsed_obj = v
          break

    opos_raw = str(parsed_obj.get("opos", 0)).lower().strip()
    oneg_raw = str(parsed_obj.get("oneg", 0)).lower().strip()

    opos = 1 if opos_raw in TRUE_VALUES else 0
    oneg = 1 if oneg_raw in TRUE_VALUES else 0

    return {
        "motivazione": str(parsed_obj.get("motivazione", ""))[:200],
        "opos": opos,
        "oneg": oneg,
    }
  raise ValueError("Risposta JSON non parsabile come dizionario")


def predict_sentiment_mistral_row(row: pd.Series) -> dict:
  user_prompt = build_unified_prompt(row, mode=MODE)
  tweet_id = str(row["id"])

  system_prompt = (
      f"{BASE_SYSTEM_PROMPT}\n\n"
      "Rispondi ESCLUSIVAMENTE con un oggetto JSON valido con questa"
      ' struttura:\n{"motivazione": "spiegazione breve", "opos": 0 o 1, "oneg":'
      " 0 o 1}"
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
              "HTTP-Referer": "[https://colab.research.google.com](https://colab.research.google.com)",
              "X-Title": f"SentiPOLC Evaluation Mistral ({MODE})",
          },
      )

      raw_text = response.choices[0].message.content or ""
      if not raw_text.strip():
        raise ValueError("Risposta API vuota")

      parsed_result = parse_json_response(raw_text)
      return {
          "motivazione": parsed_result["motivazione"],
          "opos": parsed_result["opos"],
          "oneg": parsed_result["oneg"],
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
# 6. CARICAMENTO DATASET & LOOP CON CHECKPOINT AUTOMATICO
# ==============================================================================
# Gestione flessibile per la lettura sia del CSV SentiPOLC standard che dei file Few-Shot
if MODE == "0shot" and not os.path.exists("test_set_sentipolc16_gold2000.csv"):
  df_test = pd.read_csv("sentipolc_gold_test.csv")
else:
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
new_processed_count = 0

print("=" * 80)
print(f"🚀 AVVIO ANALISI MISTRAL SMALL 24B: {STRATEGY_TYPE}")
print(f"🔹 Modello Target: [{SELECTED_MODEL}]")
print(f"🔹 Dataset Input: {INPUT_CSV} ({len(df_test)} tweet)")
print("=" * 80 + "\n")

for row in df_test.itertuples(index=False):
  row_dict = row._asdict()
  tweet_id = str(row_dict["id"])

  if tweet_id in processed_ids:
    continue

  pred = predict_sentiment_mistral_row(pd.Series(row_dict))

  res_item = {
      "id": tweet_id,
      "text": str(row_dict["text"]),
      "target_opos": int(row_dict["opos"]),
      "target_oneg": int(row_dict["oneg"]),
      "pred_opos": pred["opos"],
      "pred_oneg": pred["oneg"],
      "motivazione": pred.get("motivazione", ""),
      "raw_system_prompt": pred.get("raw_system_prompt", ""),
      "raw_prompt_user": pred.get("raw_prompt_user", ""),
      "raw_response": pred.get("raw_response", ""),
  }

  results.append(res_item)
  processed_ids.add(tweet_id)
  new_processed_count += 1

  # Stampa immediata in tempo reale
  print(
      f"[{len(results)}/{len(df_test)}] Tweet ID: {tweet_id} | Pred:"
      f" opos={pred['opos']}, oneg={pred['oneg']} (Target: opos={row_dict['opos']}, oneg={row_dict['oneg']})"
  )
  sys.stdout.flush()

  # Riepilogo parziale e salvataggio su disco ogni BATCH_SIZE (32 nuovi tweet)
  if new_processed_count % BATCH_SIZE == 0 or len(results) == len(df_test):
    elapsed = time.time() - start_time
    speed = new_processed_count / elapsed if elapsed > 0 else 0

    pd.DataFrame(results).to_csv(
        FINAL_RESULTS_CSV, index=False, encoding="utf-8"
    )

    y_true_pos = [item["target_opos"] for item in results]
    y_pred_pos = [item["pred_opos"] for item in results]
    y_true_neg = [item["target_oneg"] for item in results]
    y_pred_neg = [item["pred_oneg"] for item in results]

    acc_pos = accuracy_score(y_true_pos, y_pred_pos) * 100
    acc_neg = accuracy_score(y_true_neg, y_pred_neg) * 100
    f1_pos = f1_score(
        y_true_pos, y_pred_pos, average="macro", zero_division=0
    )
    f1_neg = f1_score(
        y_true_neg, y_pred_neg, average="macro", zero_division=0
    )
    combined_f1 = (f1_pos + f1_neg) / 2.0

    print("\n" + "=" * 80)
    print(
        f"📦 RIEPILOGO PARZIALE ({len(results)} / {len(df_test)} tweet totali |"
        f" {new_processed_count} nuovi in questa sessione)"
    )
    print(f"⏱️ Velocità reale: {speed:.2f} tweet/s")
    print(f"🔹 Accuracy opos: {acc_pos:.2f}% | Accuracy oneg: {acc_neg:.2f}%")
    print(f"-> F1-Macro opos: {f1_pos:.4f} | F1-Macro oneg: {f1_neg:.4f}")
    print(f"👉 COMBINED F1-SCORE PARZIALE: {combined_f1:.4f}")
    print("=" * 80 + "\n")
    sys.stdout.flush()

# ==============================================================================
# 7. METRICHE FINALI METICOLOSE
# ==============================================================================
if len(results) > 0:
  y_true_pos = [item["target_opos"] for item in results]
  y_pred_pos = [item["pred_opos"] for item in results]
  y_true_neg = [item["target_oneg"] for item in results]
  y_pred_neg = [item["pred_oneg"] for item in results]

  f1_pos = f1_score(y_true_pos, y_pred_pos, average="macro", zero_division=0)
  f1_neg = f1_score(y_true_neg, y_pred_neg, average="macro", zero_division=0)
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
