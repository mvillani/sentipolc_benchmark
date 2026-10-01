import json
import os
import re
import sys
import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

try:
  from openai import OpenAI
except ImportError:
  print(
      "❌ Libreria openai non installata. Esegui: !pip install -q openai pandas"
      " scikit-learn"
  )
  sys.exit(1)

# ==============================================================================
# 1. CONFIGURAZIONE MODELLO E STRATEGIA (3-Shot Pre-elaborato)
# ==============================================================================
# Inserire il modello desiderato (es. "anthropic/claude-3.5-haiku", "meta-llama/llama-3.1-8b-instruct", etc.)
SELECTED_MODEL = "anthropic/claude-haiku-4.5"
EVAL_ENGINE = "OpenRouter API"
STRATEGY_TYPE = "3-Shot Random (In-Context Learning - Seed 42 Standard)"

# Dataset di Test PRE-ELABORATO contenente sia il tweet target che i 3 esempi
INPUT_CSV = "test_set_with_random_examples_seed42.csv"
TEST_DATASET_NAME = (
    "EVALITA SentiPOLC 2016 Gold Test (with Random Examples Seed42)"
)
BATCH_SIZE = 64

model_slug = (
    SELECTED_MODEL.replace("/", "_").replace("-", "_").replace(".", "_")
)
FINAL_RESULTS_CSV = f"results_sentipolc_{model_slug}_3shot_random_seed42.csv"
FINAL_METRICS_JSON = f"metrics_sentipolc_{model_slug}_3shot_random_seed42.json"

# ==============================================================================
# 2. SETUP API KEY OPENROUTER
# ==============================================================================
try:
  from google.colab import userdata

  openrouter_api_key = userdata.get("OPENROUTER_API_KEY")
except Exception:
  openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

# ==============================================================================
# 3. STAMPA HEADER IDENTIFICATIVO
# ==============================================================================
print("=" * 80)
print("📋 RIEPILOGO SETUP ESERCIZIO SPERIMENTALE")
print("=" * 80)
print(f"🤖 Modello selezionato:  {SELECTED_MODEL}")
print(f"🌐 Aggregatore / Engine: {EVAL_ENGINE}")
print(f"🎯 Dataset di Test:      {TEST_DATASET_NAME}")
print(f"🧪 Tipologia Esercizio:  {STRATEGY_TYPE}")
print(f"📁 File Input:          {INPUT_CSV}")
print(f"💾 File Output CSV:      {FINAL_RESULTS_CSV}")
print("=" * 80 + "\n")

# ==============================================================================
# 4. SYSTEM PROMPT BASE & COSTRUZIONE FEW-SHOT DA COLONNE PRE-ELABORATE
# ==============================================================================
BASE_SYSTEM_PROMPT = """Sei un annotatore esperto di Sentiment Analysis per il dataset italiano SentiPOLC.
Per ogni tweet DEVI PRIMA spiegare brevemente la motivazione del tono e POI determinare le due polarità indipendenti opos e oneg.

- opos = 1: se sono presenti parole ed espressioni di apprezzamento, supporto, gioia o elogio, altrimenti 0.
- oneg = 1: se sono presenti parole ed espressioni di critica, attacco politico, sarcasmo, ironia o insulto, altrimenti 0.
"""


def build_few_shot_prompt_from_row(row: pd.Series) -> str:
  prompt_few_shot = "Ecco alcuni esempi guida di annotazione:\n\n"

  # Estrazione dinamica delle 3 coppie di esempi presenti nelle colonne del file
  for i in range(1, 4):
    ex_text = row.get(
        f"ex{i}_text",
        row.get(f"example{i}_text", row.get(f"ex_{i}_text", "")),
    )
    ex_opos = row.get(
        f"ex{i}_opos",
        row.get(f"example{i}_opos", row.get(f"ex_{i}_opos", 0)),
    )
    ex_oneg = row.get(
        f"ex{i}_oneg",
        row.get(f"example{i}_oneg", row.get(f"ex_{i}_oneg", 0)),
    )

    prompt_few_shot += f"Esempio {i}:\n"
    prompt_few_shot += f'Tweet: "{ex_text}"\n'
    prompt_few_shot += (
        'Risposta JSON:'
        f' {{"motivazione": "Esempio annotato", "opos": {int(ex_opos)},'
        f' "oneg": {int(ex_oneg)}}}\n\n'
    )

  target_text = str(row["text"])
  prompt_few_shot += "Ora analizza il seguente tweet target:\n"
  prompt_few_shot += f'Tweet da analizzare: "{target_text}"\n'
  prompt_few_shot += (
      "Fornisci la motivazione e le due polarita' opos e oneg in formato JSON:"
  )
  return prompt_few_shot


# ==============================================================================
# 5. INFERENZA DEDICATA VIA OPENROUTER API
# ==============================================================================
def predict_sentiment_fewshot_row(row: pd.Series) -> dict:
  user_prompt = build_few_shot_prompt_from_row(row)
  tweet_id = str(row["id"])

  system_prompt = (
      f"{BASE_SYSTEM_PROMPT}\n\n"
      "Rispondi ESCLUSIVAMENTE con un oggetto JSON valido (senza testo extra o"
      " blocchi markdown) con questa struttura:\n"
      '{"motivazione": "spiegazione breve", "opos": 0 o 1, "oneg": 0 o 1}'
  )

  max_attempts = 5
  for attempt in range(1, max_attempts + 1):
    try:
      response = client.chat.completions.create(
          model=SELECTED_MODEL,
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt},
          ],
          temperature=0.0,
          max_tokens=150,
          extra_headers={
              "HTTP-Referer": "https://colab.research.google.com",
              "X-Title": "SentiPOLC 3-Shot Evaluation",
          },
      )

      raw_text = response.choices[0].message.content.strip()

      json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
      if json_match:
        parsed_obj = json.loads(json_match.group(0))
      else:
        parsed_obj = json.loads(raw_text)

      if isinstance(parsed_obj, list) and len(parsed_obj) > 0:
        parsed_obj = parsed_obj[0]

      if isinstance(parsed_obj, dict):
        opos_val = parsed_obj.get("opos", 0)
        oneg_val = parsed_obj.get("oneg", 0)

        opos = 1 if str(opos_val).lower() in ["1", "true"] else 0
        oneg = 1 if str(oneg_val).lower() in ["1", "true"] else 0

        return {
            "motivazione": str(parsed_obj.get("motivazione", ""))[:200],
            "opos": opos,
            "oneg": oneg,
            "raw_prompt_user": user_prompt,
            "raw_system_prompt": system_prompt,
            "raw_response": raw_text,
        }

    except Exception as e:
      wait_time = attempt * 2
      print(
          f"\n⚠️ [ATTESA API] Tweet ID {tweet_id} (Tentativo"
          f" {attempt}/{max_attempts}): {type(e).__name__} - {e}. Pausa di"
          f" {wait_time}s..."
      )
      time.sleep(wait_time)

  return {
      "motivazione": "FALLIMENTO_API_DOPO_5_TENTATIVI",
      "opos": 0,
      "oneg": 0,
      "raw_prompt_user": user_prompt,
      "raw_system_prompt": system_prompt,
      "raw_response": "ERROR",
  }


# ==============================================================================
# 6. CARICAMENTO DATASET & RESTART DA CHECKPOINT
# ==============================================================================
def load_precomputed_dataset(filepath: str) -> pd.DataFrame:
  if filepath.endswith(".csv"):
    try:
      return pd.read_csv(filepath, encoding="utf-8")
    except Exception:
      return pd.read_csv(filepath, encoding="utf-8", errors="ignore")
  raise ValueError("Il file deve essere in formato CSV.")


df_test = load_precomputed_dataset(INPUT_CSV)
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
    f"🚀 Avvio/Ripresa analisi {STRATEGY_TYPE} per [{SELECTED_MODEL}] su"
    f" {len(df_test)} tweet...\n"
)

# ==============================================================================
# 7. ESECUZIONE LOOP SPERIMENTALE
# ==============================================================================
for idx, row in df_test.iterrows():
  tweet_id = str(row["id"])
  if tweet_id in processed_ids:
    continue

  tweet_text = str(row["text"])
  pred = predict_sentiment_fewshot_row(row)

  res_item = {
      "id": tweet_id,
      "text": tweet_text,
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
# 8. METRICHE FINALI E SALVATAGGIO CON GUARDRAIL
# ==============================================================================
if len(results) == 0:
  print(
      "⚠️ La lista dei risultati è vuota! Tutti i tweet sono già stati"
      " elaborati nel file CSV di output."
  )
else:
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
  print(f"⏱️ Tempo totale: {final_metrics['elapsed_time_sec']:.1f} secondi")
  print(f"🔹 Accuracy opos: {final_metrics['acc_opos']:.2f}%")
  print(f"🔹 Accuracy oneg: {final_metrics['acc_oneg']:.2f}%")
  print(f"-> F1-Macro opos: {final_metrics['f1_opos']:.4f}")
  print(f"-> F1-Macro oneg: {final_metrics['f1_neg']:.4f}")
  print(f"👉 COMBINED F1-SCORE DEFINITIVO: {final_metrics['combined_f1']:.4f}")
  print("=" * 80)
