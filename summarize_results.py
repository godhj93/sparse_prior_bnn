import json
import os
import csv
import pandas as pd

def flatten_dict(d, parent_key='', sep='_'):
    """
    중첩된 딕셔너리를 평탄화합니다.
    예: {'a': {'b': 1}} -> {'a_b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def process_json_file(file_path):
    """
    단일 JSON 파일을 처리하여 평탄화된 데이터를 반환합니다.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        info = data.get("info", {})
        
        # 기본 정보 추출
        result = {
            "model": info.get("model"),
            "type": info.get("type"),
            "data": info.get("data"),
            "sparsity": info.get("sparsity"),
            "prior_type": info.get("prior_type"),
            "std": info.get("std"),
            "moped": info.get("MOPED"),
            "pruned_dnn_accuracy": info.get("pruned_dnn_acc"),
            "pruned_dnn_nll": info.get("pruned_dnn_loss"),
            "pruned_dnn_sparsity": info.get("pruned_dnn_sparsity"),
        }
        
        # 성능 지표 추출 및 평탄화
        performance_sections = [
            "id_performance", 
            # "ood_performance", # 아래에서 별도로 처리
            "clustering_performance", 
            # "adversarial_performance" # 아래에서 별도로 처리
        ]
        
        for section in performance_sections:
            if section in data:
                flat_section = flatten_dict(data[section], parent_key=section.replace('_performance',''))
                result.update(flat_section)

        # OOD performance 동적 처리
        if "ood_performance" in data:
            ood_perf = data["ood_performance"]
            for ood_dataset_name, ood_results in ood_perf.items():
                flat_section = flatten_dict(ood_results, parent_key=f'ood_{ood_dataset_name}')
                result.update(flat_section)

        # Adversarial performance 처리: id_performance와의 차이 계산
        if "adversarial_performance" in data and "id_performance" in data:
            id_perf = data["id_performance"]
            adv_perf = data["adversarial_performance"]
            
            for attack_method, attack_results in adv_perf.items():
                # acc와 nll에 대한 절대값 차이 계산
                if "accuracy" in id_perf and "accuracy" in attack_results:
                    result[f"adversarial_{attack_method}_acc_diff"] = abs(id_perf["accuracy"] - attack_results["accuracy"])
                if "nll" in id_perf and "nll" in attack_results:
                    result[f"adversarial_{attack_method}_nll_diff"] = abs(id_perf["nll"] - attack_results["nll"])
                
                # 다른 adversarial 지표가 있다면 그대로 추가
                other_adv_metrics = {k: v for k, v in attack_results.items() if k not in ["accuracy", "nll"]}
                if other_adv_metrics:
                     flat_section = flatten_dict(other_adv_metrics, parent_key=f'adversarial_{attack_method}')
                     result.update(flat_section)
        
        return result

    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def main(args, output_csv_file):
    """
    지정된 디렉토리에서 JSON 파일을 찾아 CSV로 저장합니다.
    """
    all_results = []
    # root_directory와 그 하위 모든 디렉토리를 탐색
    for dirpath, _, filenames in os.walk(args.search_root):
        for filename in filenames:
            if filename == "best_nll_model_results2.json":
                file_path = os.path.join(dirpath, filename)
                print(f"처리 중: {file_path}")
                processed_data = process_json_file(file_path)
                if processed_data:
                    all_results.append(processed_data)

    if not all_results:
        print("처리할 JSON 파일을 찾지 못했습니다.")
        return

    # pandas DataFrame으로 변환
    output_csv_file = os.path.join(args.search_root, output_csv_file)
    df = pd.DataFrame(all_results)

    # 'sparsity'와 'std' 열에 대해 결측치를 0.0으로 채웁니다.
    # 베이스라인 모델들은 sparsity가 없으므로, 필터링 및 정렬을 위해 기본값을 설정합니다.
    if 'sparsity' in df.columns:
        df['sparsity'] = pd.to_numeric(df['sparsity'], errors='coerce').fillna(0.0)
    if 'std' in df.columns:
        df['std'] = pd.to_numeric(df['std'], errors='coerce').fillna(0.0)

    # 숫자 형식으로 변환할 열 목록
    numeric_cols = [
        'pruned_dnn_accuracy', 'pruned_dnn_nll', 'pruned_dnn_sparsity',
        'id_accuracy', 'id_nll', 'id_kld', 'clustering_silhouette',
        'clustering_davies_bouldin', 'clustering_pr', 'clustering_global_variance',
        'clustering_spearman_rho'
    ]
    # ood 및 adversarial 관련 열도 동적으로 추가
    numeric_cols.extend([col for col in df.columns if col.startswith('ood_') or col.startswith('adversarial_')])

    for col in numeric_cols:
        if col in df.columns:
            # 다른 숫자 열들은 변환할 수 없는 경우 NaN으로 유지합니다.
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 정렬 순서 적용:
    # 1. moped가 True인 행을 위로
    # 2. sparsity 값에 따라 오름차순으로 정렬
    df = df.sort_values(by=['moped', 'sparsity'], ascending=[False, True])

    # CSV 파일로 저장
    df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
    print(f"\n총 {len(all_results)}개의 결과를 '{output_csv_file}' 파일에 저장했습니다.")

    return output_csv_file

if __name__ == '__main__':
    # 검색을 시작할 상위 폴더 경로
    # 예: 'runs/cifar10/resnet20'
    import argparse
    parser = argparse.ArgumentParser(description='JSON 파일을 처리하여 결과를 CSV로 저장합니다.')
    parser.add_argument('search_root', type=str, help='검색을 시작할 상위 폴더 경로')
    
    args = parser.parse_args()
    # 결과를 저장할 CSV 파일 이름
    output_csv = 'results_summary.csv'
    
    output_csv_file = main(args, output_csv)
    
    from plot_results_paper import plot_metrics
    plot_metrics(output_csv_file)