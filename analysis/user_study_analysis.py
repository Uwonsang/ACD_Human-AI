import sys, os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import copy
from matplotlib.ticker import MaxNLocator
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from level_replay import utils

def standardize_data(data):
    standardized_data = data.copy()
    for col in data.columns:
        if 'eval' in col:
            standardized_data[col] = data[col].apply(
                lambda x: re.findall(r'\d',x)
            )
    standardized_data.set_index("Person Num", inplace=True)
    return standardized_data

def calculate_user_score(setting_data, results):
    results_data = copy.deepcopy(results)
    score_data = dict()
    score_elements_data = dict()
    score_metod_data = dict()
    person_num = 0
    for day_idx, day_data in enumerate(results):
        person_num += len(day_data)
        score_data[f"Day{day_idx}"] = dict()
        for eval_idx in range(5):
            for eval_key in ['Collaborativeness', 'Human Preference']: 
                eval_column = f'{eval_key}_eval{eval_idx}'
                eval_setting = setting_data[f"Day{day_idx+1}_eval{eval_idx}"]
                
                if eval_column not in score_data[f"Day{day_idx}"]:
                    score_data[f"Day{day_idx}"][eval_column] = {"<MAESTRO>":0, "<PLR>":0, "<Random>":0, "<OURs>":0}
                if eval_column not in score_elements_data:    
                    score_elements_data[eval_column] = {"<MAESTRO>":[], "<PLR>":[], "<Random>":[], "<OURs>":[]}
                if eval_key not in score_metod_data:
                    score_metod_data[eval_key] = {"<MAESTRO>":[], "<PLR>":[], "<Random>":[], "<OURs>":[]}
                for row_idx, row_data in enumerate(day_data[eval_column]):
                    method_dict = [eval_setting[int(i)-1] for i in row_data]
                    results_data[day_idx][eval_column][row_idx] = method_dict
                    for score_idx, method in enumerate(method_dict):
                        score = 4 - score_idx
                        score_data[f"Day{day_idx}"][eval_column][method] += score
                        score_elements_data[eval_column][method].append(score)
                        score_metod_data[eval_key][method].append(score)
    
    std_data = dict()         
    for eval_idx in range(5):
        for eval_key in ['Collaborativeness', 'Human Preference']: 
            eval_column = f'{eval_key}_eval{eval_idx}'
            if eval_column not in std_data:
                std_data[eval_column] = {"<MAESTRO>":0, "<PLR>":0, "<Random>":0, "<OURs>":0}
            for method in score_elements_data[eval_column].keys():
                std_data[eval_column][method] = np.std(score_elements_data[eval_column][method])          
    
    method_std = dict() 
    for eval_key in ['Collaborativeness', 'Human Preference']: 
        if eval_key not in method_std:
            method_std[eval_key] = {"<MAESTRO>":0, "<PLR>":0, "<Random>":0, "<OURs>":0}   
        for method in score_metod_data[eval_key].keys():
            method_std[eval_key][method] = np.std(score_metod_data[eval_key][method])
            score_metod_data[eval_key][method] = np.mean(score_metod_data[eval_key][method])
    return score_data, std_data, score_metod_data, method_std, person_num

def calculate_layout_score(score_data, person_num):
#Combine Eval_#(Eval_0, Eval_1, Eval_2, Eval_3, Eval_4) results scattered across dates (Day0, Day1, Day2) into one
#As a result, we have only one table with the two evaluations 'Collaborativeness' and 'Human Preference' summed for each Eval_#.
    layout_score_data = dict()
    avg_score_data = dict()
    score_squares_sum = dict()
    
    for day_key in score_data:
        for eval_key in score_data[day_key]:
            if eval_key not in layout_score_data:
                layout_score_data[eval_key] = {"<MAESTRO>":0, "<PLR>":0, "<Random>":0, "<OURs>":0}
                avg_score_data[eval_key] = {"<MAESTRO>":0, "<PLR>":0, "<Random>":0, "<OURs>":0}
                score_squares_sum[eval_key] = {"<MAESTRO>": 0, "<PLR>": 0, "<Random>": 0, "<OURs>": 0}
            for method_key in score_data[day_key][eval_key]:
                score = score_data[day_key][eval_key][method_key]
                layout_score_data[eval_key][method_key] += score
                avg_score_data[eval_key][method_key] += (score / person_num)
    
    return layout_score_data, avg_score_data

def calculate_total_score(layout_score_data):
#The scores divided by each experimental environment Eval_# (Eval_0, Eval_1, Eval_2, Eval_3, Eval_4) are summed by 'Collaborativeness' and 'Human Preference'.
#You should end up with a table divided into two categories: Collaborativeness and Human Preference.
    total_score_data = dict()
    for eval_key in layout_score_data:
        setting_key = eval_key.split('_')[0]
        if setting_key not in total_score_data:
            total_score_data[setting_key] = {"<MAESTRO>":0, "<PLR>":0, "<Random>":0, "<OURs>":0}
        for method_key in layout_score_data[eval_key]:
            total_score_data[setting_key][method_key] += layout_score_data[eval_key][method_key]    
    return total_score_data

def calculate_avg_score(layout_score_data):
    total_score_data = dict()
    for eval_key in layout_score_data:
        setting_key = eval_key.split('_')[0]
        if setting_key not in total_score_data:
            total_score_data[setting_key] = {"<MAESTRO>":0, "<PLR>":0, "<Random>":0, "<OURs>":0}
        for method_key in layout_score_data[eval_key]:
            total_score_data[setting_key][method_key] += (layout_score_data[eval_key][method_key]/5)    
    return total_score_data


def plot_score_graph(score_data, std_data, person_num, result_dir):
    models = ['<OURs>', '<MAESTRO>', '<PLR>', '<Random>']
    colors = ['#87ceeb', '#f08080', '#ffd700', '#90ee90']
    model_names = ['Ours', 'MAESTRO', 'PLR', 'Random']

    def prepare_data_for_matplotlib(avg_data, std_data, evals):
        scores = {model: [] for model in model_names}
        errors = {model: [] for model in model_names}
        layouts = []
        for eval in evals:
            layout = f"Test$_{eval[-1]}$"
            layouts.append(layout)
            for model, name in zip(models, model_names):
                mean = avg_data[eval][model]
                std_err = std_data[eval][model] / np.sqrt(person_num)
                scores[name].append(mean)
                errors[name].append(std_err)
        return layouts, scores, errors

    collaborativeness_evals = [f'Collaborativeness_eval{i}' for i in range(5)]
    human_preference_evals = [f'Human Preference_eval{i}' for i in range(5)]
    layouts, collab_scores, collab_errors = prepare_data_for_matplotlib(score_data, std_data, collaborativeness_evals)
    _, human_pref_scores, human_pref_errors = prepare_data_for_matplotlib(score_data, std_data, human_preference_evals)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    x = np.arange(len(layouts))
    width = 0.2

    for i, model in enumerate(model_names):
        axs[0].bar(x - width/2 + i*width, collab_scores[model], width, label=model, color=colors[i], yerr=collab_errors[model], capsize=3)
        axs[1].bar(x - width/2 + i*width, human_pref_scores[model], width, label=model, color=colors[i], yerr=human_pref_errors[model], capsize=3)

    axs[0].set_title('Collaborativeness by Layout',fontsize=18)
    axs[1].set_title('Human Preference by Layout',fontsize=18)
    axs[0].set_ylabel('Score',fontsize=15)
    for ax in axs:
        ax.set_xticks(x + width)
        ax.set_xticklabels(layouts, fontdict={'fontstyle':'italic', 'fontsize': 15})
        ax.set_ylim(0, 4)

    fig.legend(model_names, loc='lower center', ncol=4, bbox_to_anchor=(0.5,0), fontsize=15)
    # plt.subplots_adjust(bottom=0.134)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(os.path.join(result_dir, "Human_study_layout_result.png"), dpi=300)
    plt.savefig(os.path.join(result_dir, "Human_study_layout_result.pdf"))
    plt.close()


def plot_method_score_graph(score_data, std_data, person_num, result_dir):
    models = ['OURs', 'MAESTRO', 'PLR', 'Random']
    colors = ['#87ceeb', '#f08080', '#ffd700', '#90ee90']
    evals = ['Collaborativeness', 'Human Preference']
    model_names = ['Ours', 'MAESTRO', 'PLR', 'Random']
    data_frames = []

    for eval in evals:
        df_list = []
        for model in models:
            mean = score_data[eval][f'<{model}>']
            std_err = std_data[eval][f'<{model}>'] / np.sqrt(person_num)
            df_list.append({'Model': model, 'Score': mean, 'Standard Err': std_err})
        data_frames.append(pd.DataFrame(df_list))
    
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    def draw_bar_chart(ax, df, title, color_map):
        x = np.arange(len(models))
        bars = ax.bar(x, df['Score'], yerr=df['Standard Err'], align='center', ecolor='black', capsize=8, color=[color_map[model] for model in df['Model']],width=1)
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_title(title)
        ax.set_ylim(0, 4)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        for bar, score in zip(bars, df['Score']):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height()+0.3, f'{score:.2f}', ha='center', va='bottom')

    color_map = {model: color for model, color in zip(models, colors)}
    draw_bar_chart(axs[0], data_frames[0], 'Collaborativeness', color_map)
    draw_bar_chart(axs[1], data_frames[1], 'Human Preference', color_map)

    plt.tight_layout()  # 범례를 위해 조금 조정
    plt.savefig(os.path.join(result_dir, "method_score_graph.png"), dpi=300)
    plt.savefig(os.path.join(result_dir, "method_score_graph.pdf"))
    plt.close()

        
def plot_single_score_graph(total_score_data, result_dir, graph_name='total_score_graph'):
    df_total_score = pd.DataFrame(total_score_data)

    df_total_score.reset_index(inplace=True)
    df_total_score.rename(columns={'index': 'Model'}, inplace=True)

    df_melted_total_score = df_total_score.melt(id_vars=['Model'], var_name='Metric', value_name='Score')

    colors = ['skyblue', 'lightcoral', 'gold', 'lightgreen']
    sns.set_palette(sns.color_palette(colors))
    df_collaborativeness = df_melted_total_score[df_melted_total_score['Metric'] == 'Collaborativeness']
    df_human_preference = df_melted_total_score[df_melted_total_score['Metric'] == 'Human Preference']


    rc_param = {
        'axes.labelsize': 15,
        'axes.titlesize': 17,
        'legend.fontsize':12
        }
    plt.rcParams.update(rc_param)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    collab_min_score = df_collaborativeness['Score'].min() - 5
    bar_plot = sns.barplot(x='Model', y='Score', data=df_collaborativeness, ax=axes[0], palette="muted")
    axes[0].set_title('Collaborativeness Scores')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Score')
    axes[0].set_xticklabels([])
    axes[0].set_ylim(collab_min_score, None)
    
    for patch, label in zip(bar_plot.patches, df_collaborativeness['Model']):
        bar_plot.annotate(label, (patch.get_x() + patch.get_width() / 2., patch.get_height()),
                        ha = 'center', va = 'center', fontsize = 9, color = 'black', xytext = (0, 5),
                        textcoords = 'offset points')
    
    # "Human Preference" 점수 시각화
    human_pref_min_score = df_human_preference['Score'].min() - 5
    bar_plot = sns.barplot(x='Model', y='Score', data=df_human_preference, ax=axes[1], palette="muted")
    axes[1].set_title('Human Preference Scores')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].set_xticklabels([])
    axes[1].set_ylim(human_pref_min_score, None)

    for patch, label in zip(bar_plot.patches, df_human_preference['Model']):
        bar_plot.annotate(label, (patch.get_x() + patch.get_width() / 2., patch.get_height()),
                        ha = 'center', va = 'center', fontsize = 9, color = 'black', xytext = (0, 5),
                        textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{graph_name}.png"), dpi=300)
    plt.savefig(os.path.join(result_dir, f"{graph_name}.pdf"))

def main():
    base_dir = os.path.expandvars(os.path.expanduser("/app/overcooked_result/"))
    result_dir = os.path.expandvars(os.path.expanduser("/app/overcooked_result_log/user_study_graph/"))
    utils.make_dir(result_dir)
    
    day1_data = standardize_data(pd.read_csv(os.path.join(base_dir, "Day1 Data.csv")))
    day2_data = standardize_data(pd.read_csv(os.path.join(base_dir, "Day2 Data.csv")))
    # day3_data = standardize_data(pd.read_csv(os.path.join(base_dir, "Day3 Data.csv")))
    experiment_setting = pd.read_csv(os.path.join(base_dir, "Experiment Settings.csv"))
    
    # results = [day1_data, day2_data, day3_data]
    results = [day1_data, day2_data]

    score_data, std_data, metod_score_data, method_std, person_num = calculate_user_score(experiment_setting, results)
    layout_score_data, avg_score_data = calculate_layout_score(score_data, person_num)      
    total_score_data = calculate_total_score(layout_score_data)
        
    plot_score_graph(avg_score_data, std_data, person_num, result_dir)
    plot_method_score_graph(metod_score_data, method_std, person_num, result_dir)
    plot_single_score_graph(total_score_data,result_dir, graph_name='total_score_graph')
    
    avg_score_data = calculate_avg_score(layout_score_data)
    plot_single_score_graph(avg_score_data, result_dir, graph_name='avg_score_graph')
    

if __name__ == "__main__":
    main()
