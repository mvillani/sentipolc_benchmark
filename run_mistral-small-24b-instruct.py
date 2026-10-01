import json
import os
import sys
import time

BATCH_SIZE = 64

#from google.colab import userdata
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

try:
  from openai import OpenAI
except ImportError:
  print(
      '❌ Libreria openai non installata. Esegui: !pip install -q openai'
  )
  sys.exit(1)

# ==============================================================================
# 1. SETUP API KEY & SELEZIONE MODELLO SU OPENROUTER
# ==============================================================================
try:
  openrouter_api_key = userdata.get('OPENROUTER_API_KEY')
except Exception:
  openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')

if not openrouter_api_key:
  openrouter_api_key = input('⚠️ Inserisci la tua OPENROUTER_API_KEY: ')

client = OpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key=openrouter_api_key,
)

# --- DECOMMENTA IL MODELLO CHE VUOI TESTARE IN QUESTA ESECUZIONE ---
SELECTED_MODEL = 'mistralai/mistral-small-24b-instruct-2501'  # 1. Mistral Small 24B
# SELECTED_MODEL = "meta-llama/llama-3.1-8b-instruct"          # 2. Llama 3.1 8B
# SELECTED_MODEL = "meta-llama/llama-3.3-70b-instruct"         # 3. Llama 3.3 70B

# Pulizia nome file per export salvataggi
model_slug = (
    SELECTED_MODEL.replace('/', '_').replace('-', '_').replace('.', '_')
)
TEST_DATASET_PATH = 'test_set_sentipolc16_gold2000.csv'
FINAL_RESULTS_CSV = f'results_sentipolc_{model_slug}_openrouter.csv'
FINAL_METRICS_JSON = f'metrics_sentipolc_{model_slug}_openrouter.json'

# ==============================================================================
# 2. SCHEMA STRUCTURED OUTPUT & SYSTEM PROMPT
# ==============================================================================
JSON_SCHEMA = {
    'type': 'object',
    'properties': {
        'motivazione': {
            'type': 'string',
            'description': (
                'Breve analisi del testo, del tono e dell eventuale sarcasmo'
            ),
        },
        'opos': {
            'type': 'integer',
            'enum': [0, 1],
            'description': '1 se positivo o di supporto, 0 altrimenti',
        },
        'oneg': {
            'type': 'integer',
            'enum': [0, 1],
            'description': '1 se negativo, critico o sarcastico, 0 altrimenti',
        },
    },
    'required': ['motivazione', 'opos', 'oneg'],
}

SYSTEM_PROMPT = """Sei un annotatore esperto di Sentiment Analysis per il dataset italiano SentiPOLC.
Per ogni tweet DEVI PRIMA spiegare brevemente la motivazione del tono e POI determinare le due polarità indipendenti opos e oneg.

- opos = 1: se sono presenti parole ed espressioni di apprezzamento, supporto, gioia o elogio.
- oneg = 1: se sono presenti parole ed espressioni di critica, attacco politico, sarcasmo, ironia o insulto.

ESEMPI GUIDA:
1. Tweet: "Grandissimo intervento di Monti, finalmente una persona seria!"
   -> {"motivazione": "Elogio e supporto per Monti", "opos": 1, "oneg": 0}
2. Tweet: "Che bello questo governo... aumentano ancora le tasse! Bravi!"
   -> {"motivazione": "Sarcasmo e critica implicita sull aumento delle tasse", "opos": 0, "oneg": 1}
"""


# ==============================================================================
# 3. FUNZIONE INFERENZA CON VALIDAZIONE ED ERROR PRINTING
# ==============================================================================

def predict_sentiment_openrouter(text: str, tweet_id: str) -> dict:
  for attempt in range(1, 4):
    try:
      response = client.chat.completions.create(
          model=SELECTED_MODEL,
          messages=[
              {"role": "system", "content": SYSTEM_PROMPT},
              {
                  "role": "user",
                  "content": (
                      f'Tweet: "{text}"\nCompila la motivazione (max 15 parole)'
                      " e poi i valori numerici:"
                  ),
              },
          ],
          temperature=0.0,
          max_tokens=150,  # Frena il loop infinito di token!
          response_format={"type": "json_object"},
          extra_headers={
              "HTTP-Referer": "https://colab.research.google.com",
              "X-Title": "SentiPOLC Evaluation",
          },
      )

      raw_text = response.choices[0].message.content
      data = json.loads(raw_text)

      return {
          "motivazione": str(data.get("motivazione", ""))[:200],
          "opos": 1 if int(data.get("opos", 0)) >= 1 else 0,
          "oneg": 1 if int(data.get("oneg", 0)) >= 1 else 0,
      }

    except Exception as e:
      print(
          f"\n❌ [ERRORE OPENROUTER] Tweet ID {tweet_id} (Tentativo"
          f" {attempt}/3): {type(e).__name__} - {e}"
      )
      time.sleep(1.0)

  return {
      "motivazione": "FALLIMENTO_API_DOPO_3_TENTATIVI",
      "opos": 0,
      "oneg": 0,
  }
    
def predict_sentiment_openrouter_old(text: str, tweet_id: str) -> dict:
  for attempt in range(1, 4):
    try:
      response = client.chat.completions.create(
          model=SELECTED_MODEL,
          messages=[
              {'role': 'system', 'content': SYSTEM_PROMPT},
              {
                  'role': 'user',
                  'content': (
                      f'Tweet: "{text}"\nCompila la motivazione e poi i'
                      ' valori numerici:'
                  ),
              },
          ],
          temperature=0.0,
          response_format={'type': 'json_object'},
          extra_headers={
              'HTTP-Referer': 'https://colab.research.google.com',
              'X-Title': 'SentiPOLC Evaluation',
          },
      )

      raw_text = response.choices[0].message.content
      data = json.loads(raw_text)

      return {
          'motivazione': str(data.get('motivazione', '')),
          'opos': 1 if int(data.get('opos', 0)) >= 1 else 0,
          'oneg': 1 if int(data.get('oneg', 0)) >= 1 else 0,
      }

    except Exception as e:
      print(
          f'\n❌ [ERRORE OPENROUTER] Tweet ID {tweet_id} (Tentativo'
          f' {attempt}/3): {type(e).__name__} - {e}'
      )
      time.sleep(1.5)

  return {
      'motivazione': 'FALLIMENTO_API_DOPO_3_TENTATIVI',
      'opos': 0,
      'oneg': 0,
  }


# ==============================================================================
# 4. CARICAMENTO DATASET ED ESECUZIONE
# ==============================================================================
def load_sentipolc_csv(filepath: str) -> pd.DataFrame:
  rows = []
  with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      parts = line.split(',', 8)
      if len(parts) >= 9:
        tweet_id = parts[0].strip('"')
        opos = parts[2].strip('"')
        oneg = parts[3].strip('"')
        text = parts[8].strip().strip('"')
        if opos.isdigit() and oneg.isdigit():
          rows.append({
              'id': tweet_id,
              'opos': int(opos),
              'oneg': int(oneg),
              'text': text,
          })
  return pd.DataFrame(rows)


df_test = load_sentipolc_csv(TEST_DATASET_PATH)
print(f'📊 Dataset Test Caricato: {len(df_test)} tweet')

results = []
start_time = time.time()

print(
    f'🚀 Avvio analisi via OpenRouter per [{SELECTED_MODEL}] su {len(df_test)}'
    ' tweet...\n'
)

for idx, row in df_test.iterrows():
  tweet_id = str(row["id"])
  tweet_text = str(row["text"])

  # Chiamata alla funzione di predizione (es. predict_zeroshot o predict_fewshot)
  pred = predict_sentiment_openrouter(tweet_text, tweet_id)

  results.append({
      "id": tweet_id,
      "text": tweet_text,
      "target_opos": int(row["opos"]),
      "target_oneg": int(row["oneg"]),
      "pred_opos": pred["opos"],
      "pred_oneg": pred["oneg"],
      "motivazione": pred.get("motivazione", ""),
  })

  # Stampa un pallino per ogni singolo tweet elaborato
  print(".", end="", flush=True)

  # Ogni BATCH_SIZE tweet (o alla fine del dataset), calcola e mostra i risultati parziali
  if (idx + 1) % BATCH_SIZE == 0 or (idx + 1) == len(df_test):
    current_count = idx + 1
    elapsed = time.time() - start_time
    speed = current_count / elapsed

    # Calcolo metriche parziali sui tweet elaborati fino ad ora
    y_true_pos = [item["target_opos"] for item in results]
    y_pred_pos = [item["pred_opos"] for item in results]
    y_true_neg = [item["target_oneg"] for item in results]
    y_pred_neg = [item["pred_oneg"] for item in results]

    acc_pos = accuracy_score(y_true_pos, y_pred_pos) * 100
    acc_neg = accuracy_score(y_true_neg, y_pred_neg) * 100
    f1_pos = f1_score(y_true_pos, y_pred_pos, average="macro")
    f1_neg = f1_score(y_true_neg, y_pred_neg, average="macro")
    combined_f1 = (f1_pos + f1_neg) / 2.0

    print("\n" + "=" * 70)
    print(
        f"📦 PARZIALE LOTTO ({current_count} / {len(df_test)} tweet) | Velocità:"
        f" {speed:.2f} tweet/s"
    )
    print(f"🔹 Accuracy opos: {acc_pos:.2f}% | Accuracy oneg: {acc_neg:.2f}%")
    print(f"-> F1-Macro opos: {f1_pos:.4f} | F1-Macro oneg: {f1_neg:.4f}")
    print(f"👉 COMBINED F1-SCORE PARZIALE: {combined_f1:.4f}")
    print("=" * 70 + "\n")
      
# ==============================================================================
# 5. CALCOLO METRICHE FINALI E SALVATAGGIO CSV/JSON
# ==============================================================================
y_true_pos = [item['target_opos'] for item in results]
y_pred_pos = [item['pred_opos'] for item in results]
y_true_neg = [item['target_oneg'] for item in results]
y_pred_neg = [item['pred_oneg'] for item in results]

f1_pos = f1_score(y_true_pos, y_pred_pos, average='macro')
f1_neg = f1_score(y_true_neg, y_pred_neg, average='macro')
combined_f1 = (f1_pos + f1_neg) / 2.0

final_metrics = {
    'model_name': SELECTED_MODEL,
    'eval_engine': 'OpenRouter API',
    'acc_opos': accuracy_score(y_true_pos, y_pred_pos) * 100,
    'acc_oneg': accuracy_score(y_true_neg, y_pred_neg) * 100,
    'f1_opos': f1_pos,
    'f1_oneg': f1_neg,
    'combined_f1': combined_f1,
    'total_tweets': len(results),
    'elapsed_time_sec': time.time() - start_time,
}

pd.DataFrame(results).to_csv(FINAL_RESULTS_CSV, index=False, encoding='utf-8')
with open(FINAL_METRICS_JSON, 'w', encoding='utf-8') as f:
  json.dump(final_metrics, f, ensure_ascii=False, indent=2)

print('\n' + '=' * 60)
print(f'🎉 VALUTAZIONE CONCLUSO PER {SELECTED_MODEL}!')
print(f"⏱️ Tempo totale: {final_metrics['elapsed_time_sec']:.1f} secondi")
print(f"🔹 Accuracy opos: {final_metrics['acc_opos']:.2f}%")
print(f"🔹 Accuracy oneg: {final_metrics['acc_oneg']:.2f}%")
print(f"-> F1-Macro opos: {final_metrics['f1_opos']:.4f}")
print(f"-> F1-Macro oneg: {final_metrics['f1_oneg']:.4f}")
print(f"👉 COMBINED F1-SCORE DEFINITIVO: {final_metrics['combined_f1']:.4f}")
print('=' * 60)
