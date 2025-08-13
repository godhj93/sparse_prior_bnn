import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_metrics(csv_path):
    """
    Reads experiment results from a CSV file and plots performance metrics against sparsity.

    Args:
        csv_path (str): The path to the results_summary.csv file.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 데이터 타입 변환 (moped가 문자열 'True'/'False'로 읽힐 수 있음)
    if df['moped'].dtype == 'O':
         df['moped'] = df['moped'].apply(lambda x: x.strip().lower() == 'true')


    # --- 데이터 분리 ---
    # 1. Sparsity에 따라 그릴 데이터 (Normal prior, std=0.001, MOPED=False)
    main_df = df[(df['prior_type'] == 'normal') & (df['std'] < 1.0) & (df['moped'] == False)].sort_values('sparsity')

    # 2. 수평선으로 그릴 기준선(baseline) 데이터
    moped_baseline = df[(df['prior_type'] == 'normal') & (df['moped'] == True)]
    laplace_baseline = df[df['prior_type'] == 'laplace']
    normal_std1_baseline = df[(df['prior_type'] == 'normal') & (df['std'] == 1.0)]

    # --- 그릴 Metric 정의 ---
    metrics_to_plot = {
        'Pruned DNN vs BNN Acc': ('pruned_dnn_accuracy', 'id_accuracy'),
        'Pruned DNN vs BNN NLL': ('pruned_dnn_nll', 'id_nll'),
        'Accuracy': 'id_accuracy',
        'ID NLL': 'id_nll',
        'ECE': 'ood_tinyimagenet_ece',
        'OOD AUROC': 'ood_tinyimagenet_auroc_msp'
    }
    
    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
    fig.suptitle('Model Performance vs. Sparsity', fontsize=16)

    for i, (title, col_name) in enumerate(metrics_to_plot.items()):
        ax = axes[i]

        # 특별 케이스: Pruned DNN과 BNN 정확도 비교
        if isinstance(col_name, tuple):
            pruned_col, bnn_col = col_name
            # main_df에서 NaN 값을 가진 행을 제외하고 플롯
            plot_df = main_df.dropna(subset=[pruned_col, bnn_col])
            ax.plot(plot_df['sparsity'], plot_df[pruned_col], marker='x', linestyle=':', label='Pruned DNN')
            ax.plot(plot_df['sparsity'], plot_df[bnn_col], marker='o', linestyle='-', label='BNN')
        else:
            # 1. 메인 라인 플롯
            ax.plot(main_df['sparsity'], main_df[col_name], marker='o', linestyle='-', label='Normal (std=0.001)')

            # 2. 기준선(Baseline) 플롯
            if not moped_baseline.empty:
                ax.axhline(y=moped_baseline[col_name].iloc[0], color='r', linestyle='--', label=f'MOPED (val={moped_baseline[col_name].iloc[0]:.3f})')
            if not laplace_baseline.empty:
                ax.axhline(y=laplace_baseline[col_name].iloc[0], color='g', linestyle='--', label=f'Laplace (val={laplace_baseline[col_name].iloc[0]:.3f})')
            if not normal_std1_baseline.empty:
                ax.axhline(y=normal_std1_baseline[col_name].iloc[0], color='m', linestyle='--', label=f'Normal std=1.0 (val={normal_std1_baseline[col_name].iloc[0]:.3f})')

        ax.set_xlabel('Sparsity')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()

    # 그래프를 입력 CSV 파일과 동일한 디렉토리에 저장
    output_dir = os.path.dirname(csv_path)
    # 디렉토리가 없는 경우를 대비 (예: 현재 디렉토리에서 실행)
    if not output_dir:
        output_dir = '.'
    output_filename = os.path.join(output_dir, 'results_plot.png')
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    # plt.show() # 로컬에서 바로 확인하고 싶을 경우 주석 해제

    #! --- 두 번째 그림: 클러스터링 Metric ---
    clustering_metrics_to_plot = {
        'Silhouette': 'clustering_silhouette',
        'Davies-Bouldin': 'clustering_davies_bouldin',
        'PR': 'clustering_pr',
        'Global Variance': 'clustering_global_variance',
        'Spearman Rho': 'clustering_spearman_rho'
    }

    num_clustering_metrics = len(clustering_metrics_to_plot)
    fig2, axes2 = plt.subplots(1, num_clustering_metrics, figsize=(5 * num_clustering_metrics, 4))
    fig2.suptitle('Clustering Metrics vs. Sparsity', fontsize=16)

    for i, (title, col_name) in enumerate(clustering_metrics_to_plot.items()):
        ax = axes2[i]

        # 1. 메인 라인 플롯
        ax.plot(main_df['sparsity'], main_df[col_name], marker='o', linestyle='-', label='Normal (std=0.001)')

        # 2. 기준선(Baseline) 플롯
        if not moped_baseline.empty:
            ax.axhline(y=moped_baseline[col_name].iloc[0], color='r', linestyle='--', label=f'MOPED (val={moped_baseline[col_name].iloc[0]:.3f})')
        if not laplace_baseline.empty:
            ax.axhline(y=laplace_baseline[col_name].iloc[0], color='g', linestyle='--', label=f'Laplace (val={laplace_baseline[col_name].iloc[0]:.3f})')
        if not normal_std1_baseline.empty:
            ax.axhline(y=normal_std1_baseline[col_name].iloc[0], color='m', linestyle='--', label=f'Normal std=1.0 (val={normal_std1_baseline[col_name].iloc[0]:.3f})')

        ax.set_xlabel('Sparsity')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    clustering_output_filename = os.path.join(output_dir, 'clustering_results_plot.png')
    plt.savefig(clustering_output_filename)
    print(f"Clustering plot saved to {clustering_output_filename}")
    
    # --- 세 번째 그림: 적대적 공격 Metric ---
    adversarial_metrics_to_plot = {
        'FGSM NLL': 'adversarial_fgsm_nll_diff',
        'FGSM Accuracy': 'adversarial_fgsm_acc_diff',
        'PGD NLL': 'adversarial_pgd_acc_diff',
        'PGD Accuracy': 'adversarial_pgd_nll_diff'
    }

    num_adversarial_metrics = len(adversarial_metrics_to_plot)
    fig3, axes3 = plt.subplots(1, num_adversarial_metrics, figsize=(5 * num_adversarial_metrics, 4))
    fig3.suptitle('Adversarial Metrics vs. Sparsity', fontsize=16)

    for i, (title, col_name) in enumerate(adversarial_metrics_to_plot.items()):
        ax = axes3[i]

        # 1. 메인 라인 플롯
        ax.plot(main_df['sparsity'], main_df[col_name], marker='o', linestyle='-', label='Normal (std=0.001)')

        # 2. 기준선(Baseline) 플롯
        if not moped_baseline.empty:
            ax.axhline(y=moped_baseline[col_name].iloc[0], color='r', linestyle='--', label=f'MOPED (val={moped_baseline[col_name].iloc[0]:.3f})')
        if not laplace_baseline.empty:
            ax.axhline(y=laplace_baseline[col_name].iloc[0], color='g', linestyle='--', label=f'Laplace (val={laplace_baseline[col_name].iloc[0]:.3f})')
        if not normal_std1_baseline.empty:
            ax.axhline(y=normal_std1_baseline[col_name].iloc[0], color='m', linestyle='--', label=f'Normal std=1.0 (val={normal_std1_baseline[col_name].iloc[0]:.3f})')

        ax.set_xlabel('Sparsity')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    adversarial_output_filename = os.path.join(output_dir, 'adversarial_results_plot.png')
    plt.savefig(adversarial_output_filename)
    print(f"Adversarial plot saved to {adversarial_output_filename}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot model performance from a results CSV file.')
    parser.add_argument('csv_path', type=str, help='Path to the results_summary.csv file.')
    args = parser.parse_args()
    plot_metrics(args.csv_path)