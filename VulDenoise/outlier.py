import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pyod.models.abod import ABOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.so_gaal import SO_GAAL
from pyod.models.sos import SOS

# ===============================================================
FILE_PATH_AST = 'ivdetect/50/AST_loss.json'
FILE_PATH_CFG = 'ivdetect/50/CFG_loss.json'
FILE_PATH_PDG = 'ivdetect/50/PDG_loss.json'
# ===============================================================

def load_and_align_data(path_ast, path_cfg, path_pdg):
    try:
        with open(path_ast, 'r') as f: data_ast = json.load(f)
        with open(path_cfg, 'r') as f: data_cfg = json.load(f)
        with open(path_pdg, 'r') as f: data_pdg = json.load(f)
    except FileNotFoundError as e:
        print(f"error：no files - {e.filename}\nplease check the file path。")
        return None, None, None, None
    except json.JSONDecodeError as e:
        print(f"error：json format is not correct - {e}")
        return None, None, None, None

    common_sample_names = sorted(list(set(data_ast.keys()) & set(data_cfg.keys()) & set(data_pdg.keys())))
    if not common_sample_names:
        print("error：three JSON files has no common names")
        return None, None, None, None
    print(f"find {len(common_sample_names)} common samples")

    loss_ast_matrix = np.array([data_ast[name] for name in common_sample_names])
    loss_cfg_matrix = np.array([data_cfg[name] for name in common_sample_names])
    loss_pdg_matrix = np.array([data_pdg[name] for name in common_sample_names])
    
    return common_sample_names, loss_ast_matrix, loss_cfg_matrix, loss_pdg_matrix

sample_names, loss_ast, loss_cfg, loss_pdg = load_and_align_data(FILE_PATH_AST, FILE_PATH_CFG, FILE_PATH_PDG)
if sample_names is None: exit()

print(f"dataset loaded,  size: {len(sample_names)}")
print("-" * 50)


loss_vectors_concatenated = np.concatenate((loss_ast, loss_cfg, loss_pdg), axis=1)

#ablation studies
# loss_vectors_concatenated = np.concatenate((loss_ast, loss_cfg), axis=1)
# loss_vectors_concatenated = np.concatenate((loss_ast, loss_pdg), axis=1)
# loss_vectors_concatenated = np.concatenate((loss_cfg, loss_pdg), axis=1)

print("step1: loss has been concatenated")
print(f"concatenated vecs size: {loss_vectors_concatenated.shape}")
print("-" * 50)



CONTAMINATION_RATE = 0.2

ensemble_models = {
    'ABOD': ABOD(contamination=CONTAMINATION_RATE),
    'AutoEncoder': AutoEncoder(random_state=42, contamination=CONTAMINATION_RATE),
    'CBLOF': CBLOF(n_clusters=8, random_state=42, contamination=CONTAMINATION_RATE),
    'HBOS': HBOS(contamination=CONTAMINATION_RATE),
    'IForest': IForest(random_state=42, n_jobs=-1, contamination=CONTAMINATION_RATE),
    'KNN': KNN(n_neighbors=20, n_jobs=-1, contamination=CONTAMINATION_RATE),
    'LOF': LOF(n_neighbors=20, n_jobs=-1, contamination=CONTAMINATION_RATE),
    'OCSVM': OCSVM(kernel='rbf', gamma='auto', contamination=CONTAMINATION_RATE),
    'PCA': PCA(random_state=42, contamination=CONTAMINATION_RATE, n_components=0.95),
    'SO_GAAL': SO_GAAL(contamination=CONTAMINATION_RATE),
    'SOS': SOS(contamination=CONTAMINATION_RATE),
}

vote_counter = np.zeros(len(sample_names))
successful_models_count = 0

print(f"step2:{len(ensemble_models)} algorithms")
print(f"use CONTAMINATION_RATE: {CONTAMINATION_RATE}")
print("-" * 50)



print("step3: predict and vote...")
for model_name, model in ensemble_models.items():
    print(f"  -> processing: {model_name}...")
    
    try:
        model.fit(loss_vectors_concatenated)
            
        predictions = model.labels_
        noisy_indices = np.where(predictions == 1)[0]
        
        print(f"     {model_name} (contamination={CONTAMINATION_RATE}) find{len(noisy_indices)} noises。")

        if len(noisy_indices) > 0:
            vote_counter[noisy_indices] += 1
        
        successful_models_count += 1
        
    except Exception as e:
        print(f"      {model_name} processing failed: {e}")

print("-" * 50)

print("步骤四：filter noises...")
if successful_models_count > 0:
    majority_threshold = successful_models_count/2
    # majority_threshold = 9
    final_noisy_indices = np.where(vote_counter > majority_threshold)[0]
    
    print(f" {successful_models_count} algorithms run successfully。")
    print(f"threshold > {majority_threshold:.1f} ")
    print(f"\nFinally {len(final_noisy_indices)} samples has been identified noises。")

    if len(final_noisy_indices) > 0:
        results_df = pd.DataFrame({
            'sample_name': [sample_names[i] for i in final_noisy_indices],
            'votes': vote_counter[final_noisy_indices].astype(int),
            'index': final_noisy_indices
        }).sort_values(by='votes', ascending=False).reset_index(drop=True)
        
        print("\namples: (sorted by votes):")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(results_df)

        print("-" * 50)
        print("step5: saving results...")
        
        
        noisy_names_list = results_df['sample_name'].tolist()
        
        
        output_json_path = f'ivdetect/50/noisy_samples_{str(CONTAMINATION_RATE)}.json'
        
        
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(noisy_names_list, f, indent=4, ensure_ascii=False)
            print(f"succeess out_path: {output_json_path}")
        except Exception as e:
            print(f"error - {e}")

    else:
        
        print("\nno noises")

else:
    print("no algorithms run successfully")

print("-" * 50)