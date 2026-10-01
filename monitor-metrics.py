import os
import json
import glob
import pandas as pd
from sklearn.metrics import f1_score

CHECKPOINT_DIR = 'sentipolc_eval'

def evaluate_fmt2_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get("metadata", {})
    results = data.get("results", [])
    
    model_id = metadata.get("model", {}).get("model_id", "sconosciuto")
    protocol_mode = metadata.get("protocol", {}).get("mode", "sconosciuto")
    total_records = len(results)
    
    if total_records == 0:
        return None

    # Separazione record validi (status == 'ok') da quelli in errore
    valid_results = [r for r in results if r.get("status") == "ok"]
    error_count = total_records - len(valid_results)
    valid_count = len(valid_results)
    success_rate = (valid_count / total_records) * 100 if total_records > 0 else 0.0

    if valid_count == 0:
        return {
            "Model": model_id,
            "Protocol": protocol_mode,
            "Total": total_records,
            "Valid": f"0/{total_records} (0.0%)",
            "Errors": error_count,
            "F1_pos": 0.0,
            "F1_neg": 0.0,
            "F1_combined": 0.0,
            "Avg_Latency_sec": 0.0
        }

    # Estrazione vettori solo sui record validi per evitare distorsioni
    y_true_pos = [r["target_opos"] for r in valid_results]
    y_pred_pos = [r["pred_opos"] for r in valid_results]
    
    y_true_neg = [r["target_oneg"] for r in valid_results]
    y_pred_neg = [r["pred_oneg"] for r in valid_results]
    
    latencies = [r.get("latency_seconds", 0) for r in valid_results if "latency_seconds" in r]

    # Calcolo F1-score macro
    f1_pos = f1_score(y_true_pos, y_pred_pos, pos_label=1, average='binary')
    f1_neg = f1_score(y_true_neg, y_pred_neg, pos_label=1, average='binary')
    f1_combined = (f1_pos + f1_neg) / 2.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "Model": model_id,
        "Protocol": protocol_mode,
        "Total": total_records,
        "Valid": f"{valid_count}/{total_records} ({success_rate:.1f}%)",
        "Errors": error_count,
        "F1_pos": round(f1_pos, 4),
        "F1_neg": round(f1_neg, 4),
        "F1_combined": round(f1_combined, 4),
        "Avg_Latency_sec": round(avg_latency, 3)
    }

# Caricamento di tutti i file di checkpoint fmt2
json_files = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-fmt2-*.json"))
summary_data = []

for file_path in sorted(json_files):
    metrics = evaluate_fmt2_file(file_path)
    if metrics:
        summary_data.append(metrics)

# Generazione DataFrame riassuntivo
df_summary = pd.DataFrame(summary_data)

if not df_summary.empty:
    print("\n" + "=" * 100)
    print("📊 REPORT METRICHE RIASSUNTIVE (RECORD VALIDI VS ERRORI TECNICI)")
    print("=" * 100)
    print(df_summary.to_string(index=False))
else:
    print(f"⚠️ Nessun file di checkpoint trovato nella cartella '{CHECKPOINT_DIR}'.")
