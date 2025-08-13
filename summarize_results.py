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
            "pruned_dnn_nll": info.get("pruned_dnn_nll"),
            "pruned_dnn_sparsity": info.get("pruned_dnn_sparsity"),
        }
        
        # 성능 지표 추출 및 평탄화
        performance_sections = [
            "id_performance", 
            "ood_performance", 
            "clustering_performance", 
            "adversarial_performance"
        ]
        
        for section in performance_sections:
            if section in data:
                flat_section = flatten_dict(data[section], parent_key=section.replace('_performance',''))
                result.update(flat_section)
        
        return result

    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"오류: {file_path} 처리 중 문제 발생 - {e}")
        return None

def main(args, output_csv_file):
    """
    지정된 디렉토리에서 JSON 파일을 찾아 CSV로 저장합니다.
    """
    all_results = []
    # root_directory와 그 하위 모든 디렉토리를 탐색
    for dirpath, _, filenames in os.walk(args.search_root):
        for filename in filenames:
            if filename == "best_nll_model_results.json":
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

    # 'sparsity'가 비어있는 경우(NaN)를 식별하기 위해 숫자형으로 변환
    df['sparsity'] = pd.to_numeric(df['sparsity'], errors='coerce').fillna(0)

    # 정렬 순서 적용:
    # 1. sparsity가 비어있는 행을 맨 위로 (isnull()이 True인 경우)
    # 2. moped가 True인 행을 위로
    # 3. sparsity 값에 따라 오름차순으로 정렬
    df = df.sort_values(by=['moped', 'sparsity'], ascending=[False, True])


    # CSV 파일로 저장
    df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
    print(f"\n총 {len(all_results)}개의 결과를 '{output_csv_file}' 파일에 저장했습니다.")


if __name__ == '__main__':
    # 검색을 시작할 상위 폴더 경로
    # 예: 'runs/cifar10/resnet20'
    import argparse
    parser = argparse.ArgumentParser(description='JSON 파일을 처리하여 결과를 CSV로 저장합니다.')
    parser.add_argument('search_root', type=str, help='검색을 시작할 상위 폴더 경로')
    
    args = parser.parse_args()
    # 결과를 저장할 CSV 파일 이름
    output_csv = 'results_summary.csv'
    
    main(args, output_csv)