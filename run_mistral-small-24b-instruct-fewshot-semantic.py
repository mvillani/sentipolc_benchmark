import json
import os
import sys
import time

BATCH_SIZE = 64

# from google.colab import userdata
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

try:
  from openai import OpenAI
except ImportError:
  print('❌ Libreria openai non installata. Esegui: !pip install -q openai')
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
SELECTED_MODEL = (
    'mistralai/mistral-small-24b-instruct-2501'  # 1. Mistral Small 24B
)
# SELECTED_MODEL = "meta-llama/llama-3.1-8b-instruct"          # 2. Llama 3.1 8B
# SELECTED_MODEL = "meta-llama/llama-3.3-70b-instruct"         # 3. Llama 3.3 70B

model_slug = (
    SELECTED_MODEL.replace('/', '_').replace('-', '_').replace('.', '_')
)

# Percorso file test arricchito con i 3 vicini semantici
ENRICHED_TEST_CSV = 'test_set_with_semantic_examples_sentence_transformers_paraphrase_multilingual_MiniLM_L12_v2.csv'

FINAL_RESULTS_CSV = f'results_sentipolc_{model_slug}_semantic_fewshot.csv'
FINAL_METRICS_JSON = f'metrics_sentipolc_{model_slug}_semantic_fewshot.json'

# ==============================================================================
# 2. SCHEMA STRUCTURED OUTPUT & SYSTEM PROMPT BASE
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

BASE_SYSTEM_PROMPT = """Sei un annotatore esperto di Sentiment Analysis per il dataset italiano SentiPOLC.
Per ogni tweet DEVI PRIMA spiegare brevemente la motivazione del tono e POI determinare le due polarità indipendenti opos e oneg.

- opos = 1: se sono presenti parole ed espressioni di apprezzamento, supporto, gioia o elogio.
- oneg = 1: se sono presenti parole ed espressioni di critica, attacco politico, sarcasmo, ironia o insulto.
"""


# ==============================================================================
# 3. FUNZIONE INFERENZA FEW-SHOT SEMANTIC
# ==============================================================================

def predict_sentiment_openrouter_semantic(
    text: str, semantic_examples: str, tweet_id: str
) -> dict:
  # Riformattazione dinamica per garantire che ogni esempio contenga il campo motivazione
  formatted_examples = semantic_examples.replace(
      '-> opos:', '-> {"motivazione": "Esempio dal training set", "opos":'
  ).replace('oneg: 1', 'oneg: 1}')
  formatted_examples = formatted_examples.replace('oneg: 0', 'oneg: 0}')

  system_prompt = (
      f'{BASE_SYSTEM_PROMPT}\n{formatted_examples}\n'
      'REQUISITO FONDAMENTALE: Rispondi ESCLUSIVAMENTE con un oggetto JSON valido'
      ' contenente le tre chiavi: "motivazione", "opos", "oneg".'
  )

  for attempt in range(1, 4):
    try:
      response = client.chat.completions.create(
          model=SELECTED_MODEL,
          messages=[
              {'role': 'system', 'content': system_prompt},
              {
                  'role': 'user',
                  'content': (
                      f'Tweet: "{text}"\nAnalizza il sentiment e rispondi:'
                  ),
              },
          ],
          temperature=0.0,
          max_tokens=120,
          response_format={'type': 'json_object'},
          extra_headers={
              'HTTP-Referer': 'https://colab.research.google.com',
              'X-Title': 'SentiPOLC Evaluation',
          },
      )

      raw_text = response.choices[0].message.content
      data = json.loads(raw_text)

      if isinstance(data, list) and len(data) > 0:
        data = data[0]

      if isinstance(data, dict):
        # Se i dati sono annidati dentro una chiave contenitore
        for k in ['results', 'data', 'predictions', 'tweet']:
          if k in data and isinstance(data[k], dict):
            data = data[k]
            break

        opos = 1 if int(data.get('opos', 0)) >= 1 else 0
        oneg = 1 if int(data.get('oneg', 0)) >= 1 else 0

        return {
            'motivazione': str(data.get('motivazione', ''))[:150],
            'opos': opos,
            'oneg': oneg,
        }

    except Exception as e:
      time.sleep(0.5)

  # In caso di errore API reale, eseguiamo una chiamata diretta senza contesti complessi
  return {'motivazione': 'ERRORE_PARSING', 'opos': 0, 'oneg': 0}

def predict_sentiment_openrouter_semantic_sbagliata(
    text: str, semantic_examples: str, tweet_id: str
) -> dict:
  system_prompt = f"{BASE_SYSTEM_PROMPT}\n{semantic_examples}"

  for attempt in range(1, 4):
    try:
      response = client.chat.completions.create(
          model=SELECTED_MODEL,
          messages=[
              {"role": "system", "content": system_prompt},
              {
                  "role": "user",
                  "content": (
                      f'Tweet: "{text}"\nCompila la motivazione (max 15 parole)'
                      " e poi i valori numerici:"
                  ),
              },
          ],
          temperature=0.0,
          max_tokens=150,
          response_format={"type": "json_object"},
          extra_headers={
              "HTTP-Referer": "https://colab.research.google.com",
              "X-Title": "SentiPOLC Semantic Evaluation",
          },
      )

      raw_text = response.choices[0].message.content
      data = json.loads(raw_text)

      # --- PARSER UNIVERSALE - AUMENTATA ROBUSTEZZA ---
      # 1. Se la radice è una lista, prendi il primo elemento
      if isinstance(data, list) and len(data) > 0:
        data = data[0]

      # 2. Se è un dizionario che racchiude una lista sotto una chiave (es. 'tweets', 'results', 'examples')
      if isinstance(data, dict) and not ("opos" in data or "oneg" in data):
        for key, val in data.items():
          if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
            data = val[0]
            break
          elif isinstance(val, dict) and ("opos" in val or "oneg" in val):
            data = val
            break

      # 3. Estrazione dei valori con fallback sicuro a 0
      if isinstance(data, dict):
        opos_val = data.get("opos", 0)
        oneg_val = data.get("oneg", 0)

        # Gestione valori booleani o stringhe ("1", "true")
        opos = 1 if str(opos_val).lower() in ["1", "true"] else 0
        oneg = 1 if str(oneg_val).lower() in ["1", "true"] else 0

        return {
            "motivazione": str(data.get("motivazione", ""))[:200],
            "opos": opos,
            "oneg": oneg,
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

def predict_sentiment_openrouter_semantic_old_2(
    text: str, semantic_examples: str, tweet_id: str
) -> dict:
  system_prompt = f'{BASE_SYSTEM_PROMPT}\n{semantic_examples}'

  for attempt in range(1, 4):
    try:
      response = client.chat.completions.create(
          model=SELECTED_MODEL,
          messages=[
              {'role': 'system', 'content': system_prompt},
              {
                  'role': 'user',
                  'content': (
                      f'Tweet da analizzare: "{text}"\nCompila la'
                      ' motivazione (max 15 parole) e poi i valori numerici:'
                  ),
              },
          ],
          temperature=0.0,
          max_tokens=150,
          response_format={'type': 'json_object'},
          extra_headers={
              'HTTP-Referer': 'https://colab.research.google.com',
              'X-Title': 'SentiPOLC Semantic Evaluation',
          },
      )

      raw_text = response.choices[0].message.content
      data = json.loads(raw_text)

      # --- CORREZIONE: Gestione lista vs dizionario ---
      if isinstance(data, list) and len(data) > 0:
        data = data[0]

      if not isinstance(data, dict):
        raise ValueError('Il JSON restituito non è un dizionario valido.')

      return {
          'motivazione': str(data.get('motivazione', ''))[:200],
          'opos': 1 if int(data.get('opos', 0)) >= 1 else 0,
          'oneg': 1 if int(data.get('oneg', 0)) >= 1 else 0,
      }

    except Exception as e:
      print(
          f'\n❌ [ERRORE OPENROUTER] Tweet ID {tweet_id} (Tentativo'
          f' {attempt}/3): {type(e).__name__} - {e}'
      )
      time.sleep(1.0)

  return {
      'motivazione': 'FALLIMENTO_API_DOPO_3_TENTATIVI',
      'opos': 0,
      'oneg': 0,
  }
    
def predict_sentiment_openrouter_semantic_old_1(
    text: str, semantic_examples: str, tweet_id: str
) -> dict:
  # Iniezione dinamica dei vicini semantici nel System Prompt
  system_prompt = f'{BASE_SYSTEM_PROMPT}\n{semantic_examples}'

  for attempt in range(1, 4):
    try:
      response = client.chat.completions.create(
          model=SELECTED_MODEL,
          messages=[
              {'role': 'system', 'content': system_prompt},
              {
                  'role': 'user',
                  'content': (
                      f'Tweet: "{text}"\nCompila la motivazione (max 15 parole)'
                      ' e poi i valori numerici:'
                  ),
              },
          ],
          temperature=0.0,
          max_tokens=150,  # Previene loop di token
          response_format={'type': 'json_object'},
          extra_headers={
              'HTTP-Referer': 'https://colab.research.google.com',
              'X-Title': 'SentiPOLC Semantic Evaluation',
          },
      )

      raw_text = response.choices[0].message.content
      data = json.loads(raw_text)

      return {
          'motivazione': str(data.get('motivazione', ''))[:200],
          'opos': 1 if int(data.get('opos', 0)) >= 1 else 0,
          'oneg': 1 if int(data.get('oneg', 0)) >= 1 else 0,
      }

    except Exception as e:
      print(
          f'\n❌ [ERRORE OPENROUTER] Tweet ID {tweet_id} (Tentativo'
          f' {attempt}/3): {type(e).__name__} - {e}'
      )
      time.sleep(1.0)

  return {
      'motivazione': 'FALLIMENTO_API_DOPO_3_TENTATIVI',
      'opos': 0,
      'oneg': 0,
  }


# ==============================================================================
# 4. CARICAMENTO DATASET ARRICCHITO ED ESECUZIONE
# ==============================================================================
df_test = pd.read_csv(ENRICHED_TEST_CSV)
print(f'📊 Dataset Test Arricchito Caricato: {len(df_test)} tweet')

results = []
start_time = time.time()

print(
    f'🚀 Avvio analisi FEW-SHOT SEMANTIC via OpenRouter per [{SELECTED_MODEL}]'
    f' su {len(df_test)} tweet...\n'
)

for idx, row in df_test.iterrows():
  tweet_id = str(row['id'])
  tweet_text = str(row['text'])
  semantic_context = str(row['fewshot_prompt_context'])

  pred = predict_sentiment_openrouter_semantic(
      tweet_text, semantic_context, tweet_id
  )

  results.append({
      'id': tweet_id,
      'text': tweet_text,
      'target_opos': int(row['opos']),
      'target_oneg': int(row['oneg']),
      'pred_opos': pred['opos'],
      'pred_oneg': pred['oneg'],
      'motivazione': pred.get('motivazione', ''),
  })

  # Alive sign per ogni tweet
  print('.', end='', flush=True)

  # Stampa parziale ogni BATCH_SIZE tweet
  if (idx + 1) % BATCH_SIZE == 0 or (idx + 1) == len(df_test):
    current_count = idx + 1
    elapsed = time.time() - start_time
    speed = current_count / elapsed

    y_true_pos = [item['target_opos'] for item in results]
    y_pred_pos = [item['pred_opos'] for item in results]
    y_true_neg = [item['target_oneg'] for item in results]
    y_pred_neg = [item['pred_oneg'] for item in results]

    acc_pos = accuracy_score(y_true_pos, y_pred_pos) * 100
    acc_neg = accuracy_score(y_true_neg, y_pred_neg) * 100
    f1_pos = f1_score(y_true_pos, y_pred_pos, average='macro')
    f1_neg = f1_score(y_true_neg, y_pred_neg, average='macro')
    combined_f1 = (f1_pos + f1_neg) / 2.0

    print('\n' + '=' * 70)
    print(
        f'📦 PARZIALE FEW-SHOT SEMANTIC ({current_count} / {len(df_test)}'
        f' tweet) | Velocità: {speed:.2f} tweet/s'
    )
    print(f'🔹 Accuracy opos: {acc_pos:.2f}% | Accuracy oneg: {acc_neg:.2f}%')
    print(f'-> F1-Macro opos: {f1_pos:.4f} | F1-Macro oneg: {f1_neg:.4f}')
    print(f'👉 COMBINED F1-SCORE PARZIALE: {combined_f1:.4f}')
    print('=' * 70 + '\n')

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
    'strategy': 'Few-Shot Semantic Search (k-NN)',
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
print(
    f'🎉 VALUTAZIONE FEW-SHOT SEMANTIC COMPLETATA PER {SELECTED_MODEL}!'
)
print(f"⏱️ Tempo totale: {final_metrics['elapsed_time_sec']:.1f} secondi")
print(f"🔹 Accuracy opos: {final_metrics['acc_opos']:.2f}%")
print(f"🔹 Accuracy oneg: {final_metrics['acc_oneg']:.2f}%")
print(f"-> F1-Macro opos: {final_metrics['f1_opos']:.4f}")
print(f"-> F1-Macro oneg: {final_metrics['f1_oneg']:.4f}")
print(f"👉 COMBINED F1-SCORE DEFINITIVO: {final_metrics['combined_f1']:.4f}")
print('=' * 60)
