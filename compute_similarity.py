import numpy as np
import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt
import os
import json
import sys
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from tqdm import tqdm
from collections import defaultdict

MODELS = ['resnet18', 'resnet34', 'dinov2', 'clip']
DATASETS = ['simple1K', 'Paris_val', 'VOC_val']

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
img_results_dir = os.path.join(results_dir, 'imgresults')
gen_results_dir = os.path.join(results_dir, 'genresults')
results_dict_file = os.path.join(gen_results_dir, 'evaluation_results.json')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def load_results_dict():
    if os.path.exists(results_dict_file):
        with open(results_dict_file, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_results_dict(results_dict):
    with open(results_dict_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, cls=NumpyEncoder)

def calculate_map_simple(similarity_matrix, files, k=10):
    ap_values = []
    class_precisions = defaultdict(list)
    class_recalls = defaultdict(list)
      
    # Eliminar esta línea que limita a 100 consultas
    # max_queries = min(100, len(files))
    valid_queries = []
    
    # Recoger todas las imágenes con clase conocida como consultas
    for i in range(len(files)):
        if len(files[i]) > 1:  # Si tiene clase
            valid_queries.append(i)
            # Eliminar este if que limita a 100
            # if len(valid_queries) >= max_queries:
            #     break
    
    if not valid_queries:
        return 0.0, {}
    
    for query_idx in valid_queries:
        try:
            query_class = files[query_idx][1]
            similarities = similarity_matrix[query_idx]
            sorted_indices = np.argsort(-similarities)
            retrieval_indices = [idx for idx in sorted_indices if idx != query_idx][:k]
            
            relevance = []
            for idx in retrieval_indices:
                if idx < len(files) and len(files[idx]) > 1 and files[idx][1] == query_class:
                    relevance.append(1)
                else:
                    relevance.append(0)
            
            total_relevant = -1
            for f in files:
                if len(f) > 1 and f[1] == query_class:
                    total_relevant += 1
            
            if total_relevant == 0:
                continue
            
            precision_values = []
            recall_values = []
            num_relevant_found = 0
            
            for i, rel in enumerate(relevance):
                if rel == 1:
                    num_relevant_found += 1
                
                precision = num_relevant_found / (i + 1)
                recall = num_relevant_found / total_relevant
                
                precision_values.append(precision)
                recall_values.append(recall)
                
                if rel == 1:
                    ap_values.append(precision)
                
                class_precisions[query_class].append(precision)
                class_recalls[query_class].append(recall)
                
        except Exception:
            continue
    
    mAP = np.mean(ap_values) if ap_values else 0.0
    
    standard_recalls = np.linspace(0, 1, 11)
    interpolated_data = {}
    
    for class_name in class_precisions:
        recalls = np.array(class_recalls[class_name])
        precisions = np.array(class_precisions[class_name])
        
        sorted_indices = np.argsort(recalls)
        recalls = recalls[sorted_indices]
        precisions = precisions[sorted_indices]
        
        interpolated_precisions = []
        for recall_level in standard_recalls:
            mask = recalls >= recall_level
            if np.any(mask):
                interpolated_precisions.append(float(np.max(precisions[mask])))
            else:
                interpolated_precisions.append(0.0)
        
        interpolated_data[class_name] = {
            'recalls': standard_recalls.tolist(),
            'precisions': interpolated_precisions
        }
    
    if interpolated_data:
        avg_precisions = np.zeros(11)
        for class_name in interpolated_data:
            avg_precisions += np.array(interpolated_data[class_name]['precisions'])
        avg_precisions /= len(interpolated_data)
        
        interpolated_data['average'] = {
            'recalls': standard_recalls.tolist(),
            'precisions': avg_precisions.tolist()
        }
    
    return mAP, interpolated_data

def process_dataset_model(dataset, model, force_recompute=False):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'T2images', dataset)
    image_dir = os.path.join(data_dir, 'images')
    list_of_images = os.path.join(data_dir, 'list_of_images.txt')
    feat_file = os.path.join('data', f'feat_{model}_{dataset}.npy')
    
    output_dir = os.path.join(img_results_dir, f"{dataset}_{model}")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(img_results_dir, exist_ok=True)
    os.makedirs(gen_results_dir, exist_ok=True)
    
    if os.path.exists(output_dir) and os.listdir(output_dir):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(img_results_dir, f"{dataset}_{model}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        if not os.path.exists(feat_file) or not os.path.exists(list_of_images):
            return False
        
        with open(list_of_images, "r") as file: 
            files = [f.strip().split('\t') for f in file]
            
        feats = np.load(feat_file)
        
        results_dict = load_results_dict()
        key = f"{dataset}_{model}"
        
        if key in results_dict and not force_recompute:
            return True
        
        similarity = cosine_similarity(feats)
        dist = 1 - similarity
        dist_idx = np.argsort(dist, axis=1)
        
        k = 10
        iteration_results = []
        
        for iteration in range(10):
            query = np.random.permutation(dist.shape[0])[0]
            best_idx = dist_idx[query, :k+1]
            
            query_class = files[query][1] if len(files[query]) > 1 else "desconocida"
            
            matching_classes = 0
            for idx in best_idx:
                if idx < len(files) and len(files[idx]) > 1 and files[idx][1] == query_class:
                    matching_classes += 1
            
            result = {
                'iteration': iteration,
                'query_idx': int(query),
                'query_class': query_class,
                'best_idx': best_idx.tolist() if isinstance(best_idx, np.ndarray) else best_idx,
                'matching_classes': int(matching_classes-1)
            }
            iteration_results.append(result)
        
        sorted_iterations = sorted(iteration_results, key=lambda x: x['matching_classes'], reverse=True)
        
        best_iterations = sorted_iterations[:5]
        worst_iterations = sorted_iterations[-5:]
        
        def visualize_iteration(iteration_data, title, filename):
            query_idx = iteration_data['query_idx']
            best_idx = iteration_data['best_idx']
            query_class = iteration_data['query_class']
            
            fig, ax = plt.subplots(1, k+1, figsize=(15, 3))
            fig.suptitle(f"{title}: {iteration_data['matching_classes']}/{k}")
            
            for i, idx in enumerate(best_idx):
                try:
                    img_filename = os.path.join(image_dir, files[idx][0])
                    im = io.imread(img_filename)
                    im = transform.resize(im, (64, 64))
                    ax[i].imshow(im)
                    ax[i].set_axis_off()
                    
                    if len(files[idx]) > 1:
                        img_class = files[idx][1]
                        match = "✓" if img_class == query_class else "✗"
                        ax[i].set_title(f"{img_class} {match}")
                except Exception:
                    ax[i].text(0.5, 0.5, "Error", ha='center', va='center')
                    ax[i].set_axis_off()
            
            ax[0].patch.set(lw=6, ec='b')
            ax[0].set_axis_on()
            
            plt.tight_layout()
            img_path = os.path.join(output_dir, filename)
            plt.savefig(img_path)
            plt.close()
            return img_path
        
        best_image_paths = []
        for i, iteration_data in enumerate(best_iterations):
            title = f"TOP {i+1}: Best classification"
            filename = f"mejor_clasificacion_{i+1}.png"
            img_path = visualize_iteration(iteration_data, title, filename)
            best_image_paths.append(img_path)
        
        worst_image_paths = []
        for i, iteration_data in enumerate(worst_iterations):
            title = f"BOTTOM {i+1}: Peor clasificación"
            filename = f"peor_clasificacion_{i+1}.png"
            img_path = visualize_iteration(iteration_data, title, filename)
            worst_image_paths.append(img_path)
        
        try:
            mAP, interpolated_pr_data = calculate_map_simple(similarity, files, k=k)
            
            results_dict[key] = {
                'dataset': dataset,
                'model': model,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'mAP': float(mAP),
                'best_iterations': best_iterations,
                'worst_iterations': worst_iterations,
                'best_image_paths': best_image_paths,
                'worst_image_paths': worst_image_paths,
                'output_dir': output_dir,
                'precision_recall': {
                    'standard_recalls': np.linspace(0, 1, 11).tolist(),
                    'class_data': interpolated_pr_data
                }
            }
            
            save_results_dict(results_dict)
        except Exception:
            total_matches = 0
            total_queries = 0
            for result in iteration_results:
                matches = result['matching_classes']
                total_matches += matches
                total_queries += 1
                
            mAP_approx = total_matches / (total_queries * k) if total_queries > 0 else 0
            
            results_dict[key] = {
                'dataset': dataset,
                'model': model,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'mAP': float(mAP_approx),
                'mAP_method': 'approximated',
                'best_iterations': best_iterations,
                'worst_iterations': worst_iterations,
                'best_image_paths': best_image_paths,
                'worst_image_paths': worst_image_paths,
                'output_dir': output_dir
            }
            
            save_results_dict(results_dict)
        
        return True
            
    except FileNotFoundError:
        return False
    except Exception:
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(img_results_dir, exist_ok=True)
    os.makedirs(gen_results_dir, exist_ok=True)
    
    parser = argparse.ArgumentParser(description='Calcular similitud y evaluar modelos')
    parser.add_argument('--dataset', choices=DATASETS, help='Dataset específico a procesar')
    parser.add_argument('--model', choices=MODELS, help='Modelo específico a usar')
    parser.add_argument('--force', action='store_true', help='Forzar recálculo aunque ya existan resultados')
    args = parser.parse_args()
    
    if args.dataset and args.model:
        process_dataset_model(args.dataset, args.model, force_recompute=args.force)
    elif args.dataset:
        for model in MODELS:
            process_dataset_model(args.dataset, model, force_recompute=args.force)
    elif args.model:
        for dataset in DATASETS:
            process_dataset_model(dataset, args.model, force_recompute=args.force)
    else:
        for dataset in DATASETS:
            for model in MODELS:
                success = process_dataset_model(dataset, model, force_recompute=args.force)