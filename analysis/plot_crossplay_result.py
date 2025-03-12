import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

def get_result_files(data_dir, score_type="dense", model_prefix="scientific_reports_"):
    model_names = [
        f"{model_prefix}3.pbt_return",
        f"{model_prefix}3.pbt_td",
        f"{model_prefix}2.plr_td",
        f"{model_prefix}1.random_random",
    ]
    
    layout_result_scores = np.zeros((5, len(model_names), len(model_names)+1))
    layout_max_score = np.zeros(5)
    
    for i in range(len(model_names)):
        for j in range(i, len(model_names)):
            if score_type == "sparse":
                result_file_name = model_names[j] + " vs " + model_names[i] + "_" + score_type + ".csv"
            else:
                result_file_name = model_names[j] + " vs " + model_names[i] + ".csv"
            result_path = os.path.join(data_dir, result_file_name)
            df_result = pd.read_csv(result_path)
            #mean행 가져오기
            mean_row = df_result.iloc[1][1:]
            
            #mean행의 평균 값 구하기
            mean_score = mean_row.mean()
            
            for k in range(5):
                if layout_max_score[k] < mean_row[k]:
                    layout_max_score[k] = mean_row[k]  
                layout_result_scores[k, i, j] = mean_row[k]
                layout_result_scores[k, j, i] = mean_row[k]
                
            
    
    for i in range(5):
        for j in range(len(model_names)):
            layout_result_scores[i,j, len(model_names)] = np.mean(layout_result_scores[i, j, :-1])
    
    return layout_result_scores, layout_max_score

def normalize_score(score_data, min_max_score):
    normalize_score = np.zeros(score_data.shape)
    for i in range(len(score_data)):
        max_score = np.max(score_data[i])
        normalize_score[i] = score_data[i] / max_score 
                                                         
    normalize_avg_score = np.mean(normalize_score, axis=0)
    return normalize_score, normalize_avg_score

def plot_heatmap(score_data, result_dir, title="", save_name= "", cmap ="BuPu"):
    labels = ["Ours", "MEASTRO", "PLR", "Random", "mean"]
    
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(score_data, annot=True, fmt=".2f", cmap=cmap, xticklabels=labels, yticklabels=labels[:-1], annot_kws={"size": 18})
    plt.title(title, fontsize=20, fontdict={'fontstyle':'italic'})
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18) 
    cbar.locator = MultipleLocator(0.2)
    cbar.update_ticks()
    if save_name == "":
        save_name = title
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{save_name}_crossplay_result.png"), dpi=300)
    plt.savefig(os.path.join(result_dir, f"{save_name}_crossplay_result.pdf"))
    plt.close()
    
    
    
def main():
    data_dir = os.path.expandvars(os.path.expanduser("/app/overcooked_result_log/lr-20250226-072933- Ego_Co_player/"))    
    result_dir = os.path.expandvars(os.path.expanduser("/app/overcooked_result_log/crossplay_graph/"))
    model_folder_name = "scientific_reports_"
    
    os.makedirs(result_dir, exist_ok=True)
    
    layout_result_scores, layout_max_score = get_result_files(data_dir, score_type="dense", model_prefix=model_folder_name)
    normalized_score, normalized_avg_score = normalize_score(layout_result_scores, layout_max_score)
    for test_idx, score_data in enumerate(normalized_score):
        plot_heatmap(score_data, result_dir, f'Test$_{test_idx}$', f'Test{test_idx}')
        
    plot_heatmap(normalized_avg_score, result_dir, "Mean", cmap="bone_r")
    # plot_heatmap(normalized_score, result_dir)
    
    

if __name__ == "__main__":
    main()