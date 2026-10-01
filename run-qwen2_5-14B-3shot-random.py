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
MODE = "random"  # <-- Imposta "0shot", "random", oppure "semantic"
SELECTED_MODEL = "qwen/qwen-2.5-14b-instruct"
EVAL_ENGINE = "OpenRouter API"
BATCH_SIZE = 32

model_slug = (
    SELECTED_MODEL.replace("/", "_").replace("-", "_").replace(".", "_")
)

# Gestione dinamica dei file di input e output
if MODE == "0shot":
  STRATEGY_TYPE = "0-Shot Pure (Zero-Shot Text Prompting)"
  INPUT_CSV = "sentipolc_gold_test.csv"
  TEST_DATASET_NAME = "EVALITA SentiPOLC 2016 Gold Test"
  FINAL_RESULTS_CSV = f"results_sentipolc_{model_slug}_0shot_pure.csv"
  FINAL_METRICS_JSON = f"metrics_sentipolc_{model_slug}_0shot_pure.json"
  LOG_FILE = f"api_response_errors_0shot_{model_slug}.log"

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
# 2. SETUP LOGGING DEDICATO
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
# 4. SYSTEM PROMPT TESTUALE & COSTRUTTORE USER PROMPT
# ==============================================================================
BASE_SYSTEM_PROMPT = """Sei un annotatore esperto di Sentiment Analysis per il dataset italiano SentiPOLC.
Per ogni tweet analizzato DEVI stampare ESCLUSIVAMENTE 3 righe con questo formato esatto:

MOTIVAZIONE: breve spiegazione della polarità
OPOS: 1 oppure 0
ONEG: 1 oppure 0

Criteri di assegnazione:
- OPOS: 1 se sono presenti parole ed espressioni di apprezzamento, supporto, gioia o elogio, altrimenti 0.
- ONEG: 1 se sono presenti parole ed espressioni di critica, attacco politico, sarcasmo, ironia o insulto, altrimenti 0."""


def build_text_prompt(row: pd.Series, mode: str) -> str:
  target_text = str(row["text"])

  if mode == "0shot":
    return f'Analizza il seguente tweet:\nTweet: "{target_text}"'

  elif mode == "semantic":
    context = row.get("fewshot_prompt_context", "")
    return (
        f"{context}\n"
        "Ora analizza il seguente tweet target applicando i criteri sopra:\n"
        f'Tweet: "{target_text}"'
    )

  elif mode == "random":
    fewshot_str = "Ecco alcuni esempi guida di annotazione SentiPOLC:\n\n"
    for i in range(1, 4):
      ex_text = row.get(f"ex{i}_text", "")
      ex_opos = row.get(f"ex{i}_opos", 0)
      ex_oneg = row.get(f"ex{i}_oneg", 0)
      fewshot_str += (
          f'Esempio {i}:\nTweet: "{ex_text}"\nMOTIVAZIONE: Esempio'
          f" guida\nOPOS: {ex_opos}\nONEG: {ex_oneg}\n\n"
      )

    return (
        f"{fewshot_str}Ora analizza il seguente tweet target applicando gli"
        f' stessi criteri:\nTweet: "{target_text}"'
    )


# ==============================================================================
# 5. ENGINE DI INFERENZA TESTUALE E PARSER REGEX
# ==============================================================================
def parse_text_response(raw_text: str) -> dict:
  """Estrae opos e oneg da testo libero o da formato riga per riga."""
  opos_match = re.search(r"opos\s*[:=]\s*([01])", raw_text, re.IGNORECASE)
  oneg_match = re.search(r"oneg\s*[:=]\s*([01])", raw_text, re.IGNORECASE)

  if opos_match and oneg_match:
    return {
        "motivazione": raw_text.split("\n")[0][:200],
        "opos": int(opos_match.group(1)),
        "oneg": int(oneg_match.group(1)),
    }
  raise ValueError("Estrazione regex fallita sul testo fornito dall'API")


def predict_sentiment_unified_row(row: pd.Series) -> dict:
  user_prompt = build_text_prompt(row, mode=MODE)
  tweet_id = str(row["id"])

  max_attempts = 5
  for attempt in range(1, max_attempts + 1):
    raw_text = ""
    try:
      response = client.chat.completions.create(
          model=SELECTED_MODEL,
          messages=[
              {"role": "system", "content": BASE_SYSTEM_PROMPT},
              {"role": "user", "content": user_prompt},
          ],
          temperature=0.0,
          max_tokens=120,
          extra_headers={
              "HTTP-Referer": "https://colab.research.google.com",
              "X-Title": f"SentiPOLC Evaluation ({MODE})",
          },
      )

      raw_text = response.choices[0].message.content or ""
      if not raw_text.strip():
        raise ValueError("Risposta API vuota")

      parsed_result = parse_text_response(raw_text)
      return {
          "motivazione": parsed_result["motivazione"],
          "opos": parsed_result["opos"],
          "oneg": parsed_result["oneg"],
          "raw_prompt_user": user_prompt,
          "raw_system_prompt": BASE_SYSTEM_PROMPT,
          "raw_response": raw_text,
      }

    except Exception as e:
      error_msg = (
          f"TWEET_ID: {tweet_id} | TENTATIVO: {attempt}/{max_attempts} | ERRORE:"
          f" {type(e).__name__}: {e}\n"
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
      "raw_system_prompt": BASE_SYSTEM_PROMPT,
      "raw_response": "ERROR",
  }


# ==============================================================================
# 6. CARICAMENTO DATASET & LOOP DI ESECUZIONE
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
new_processed_count = 0

print("=" * 80)
print(f"🚀 AVVIO ANALISI TESTUALE: {STRATEGY_TYPE}")
print(f"🔹 Modello Target: [{SELECTED_MODEL}]")
print(f"🔹 Dataset Input: {INPUT_CSV} ({len(df_test)} tweet)")
print("=" * 80 + "\n")

for row in df_test.itertuples(index=False):
  row_dict = row._asdict()
  tweet_id = str(row_dict["id"])

  if tweet_id in processed_ids:
    continue

  pred = predict_sentiment_unified_row(pd.Series(row_dict))

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

  # Riepilogo parziale ogni BATCH_SIZE (32 nuovi tweet)
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
# 7. METRICHE FINALI
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
