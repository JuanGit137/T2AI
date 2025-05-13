import json
import matplotlib.pyplot as plt
import os
import numpy as np
import skimage.io as io
import skimage.transform as transform
import datetime
from sklearn.metrics.pairwise import cosine_similarity

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, 'results')
img_results_dir = os.path.join(results_dir, 'imgresults')
graphs_results_dir = os.path.join(results_dir, 'graphsresults')
gen_results_dir = os.path.join(results_dir, 'genresults')

os.makedirs(results_dir, exist_ok=True)
os.makedirs(img_results_dir, exist_ok=True)
os.makedirs(graphs_results_dir, exist_ok=True)
os.makedirs(gen_results_dir, exist_ok=True)

with open("results/genresults/evaluation_results.json", "r") as f:
    data = json.load(f)
print("Do you want to:")
print("1. Generate mAP and PR graphs? (Enter 1)")
print("2. Generate images of the 5 best and 5 worst examples? (Enter 2)")
a = int(input("Enter your choice: "))
    
if a == 1:    
    datasets = {}
    for key, val in data.items():
        dataset = val["dataset"]
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(val)

    for dataset, models_data in datasets.items():
        model_names = [m['model'] for m in models_data]
        mAPs = [m['mAP'] for m in models_data]

        plt.figure(figsize=(8, 5))
        plt.bar(model_names, mAPs, color='skyblue', width=0.4)
        plt.title(f'mAP Comparison - {dataset}')
        plt.ylabel('Mean Average Precision')
        plt.xlabel('Model')
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        map_graph_path = os.path.join(graphs_results_dir, f'{dataset}_map_comparison.png')
        plt.savefig(map_graph_path)
        plt.close()

        plt.figure(figsize=(8, 5))
        for m in models_data:
            if 'precision_recall' in m and 'class_data' in m['precision_recall'] and 'average' in m['precision_recall']['class_data']:
                pr_data = m['precision_recall']['class_data']['average']
                recalls = pr_data['recalls']
                precisions = pr_data['precisions']
                plt.plot(recalls, precisions, label=m['model'])

        plt.title(f'Average Precision-Recall Curve - {dataset}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        pr_graph_path = os.path.join(graphs_results_dir, f'{dataset}_pr_curve.png')
        plt.savefig(pr_graph_path)
        plt.close()
        
        print(f"Graphs for {dataset} saved to {graphs_results_dir}")

elif a == 2:
    def load_results_dict(results_dict_file):
        if os.path.exists(results_dict_file):
            with open(results_dict_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}

    def generate_visualization_images(dataset, model, similarity, files, dist_idx, output_dir, image_dir=None):
        if image_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            image_dir = os.path.join(base_dir, 'T2images', dataset, 'images')
        
        k = 10
        iteration_results = []
        
        for iteration in range(10):
            query = np.random.permutation(similarity.shape[0])[0]
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
        
        return {
            'best_iterations': best_iterations,
            'worst_iterations': worst_iterations,
            'best_image_paths': best_image_paths,
            'worst_image_paths': worst_image_paths
        }

    def regenerate_images_from_results(dataset, model, force_regenerate=False):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, 'results')
        img_results_dir = os.path.join(results_dir, 'imgresults')
        gen_results_dir = os.path.join(results_dir, 'genresults')
        results_dict_file = os.path.join(gen_results_dir, 'evaluation_results.json')
        
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(img_results_dir, exist_ok=True)
        os.makedirs(gen_results_dir, exist_ok=True)
        
        data_dir = os.path.join(base_dir, 'T2images', dataset)
        image_dir = os.path.join(data_dir, 'images')
        list_of_images = os.path.join(data_dir, 'list_of_images.txt')
        feat_file = os.path.join('data', f'feat_{model}_{dataset}.npy')
        
        output_dir = os.path.join(img_results_dir, f"{dataset}_{model}")
        
        if force_regenerate and os.path.exists(output_dir):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(img_results_dir, f"{dataset}_{model}_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            if not os.path.exists(feat_file) or not os.path.exists(list_of_images):
                print(f"Required files not found for {dataset}_{model}")
                return False
            
            with open(list_of_images, "r") as file: 
                files = [f.strip().split('\t') for f in file]
                
            feats = np.load(feat_file)
            
            similarity = cosine_similarity(feats)
            dist = 1 - similarity
            dist_idx = np.argsort(dist, axis=1)
            
            visualization_results = generate_visualization_images(
                dataset, model, similarity, files, dist_idx, output_dir, image_dir
            )
            
            results_dict = load_results_dict(results_dict_file)
            key = f"{dataset}_{model}"
            
            if key in results_dict:
                results_dict[key].update({
                    'best_iterations': visualization_results['best_iterations'],
                    'worst_iterations': visualization_results['worst_iterations'],
                    'best_image_paths': visualization_results['best_image_paths'],
                    'worst_image_paths': visualization_results['worst_image_paths'],
                    'output_dir': output_dir,
                })
            
            with open(results_dict_file, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, cls=NumpyEncoder)
                
            print(f"Successfully generated images for {dataset}_{model}")
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error generating images for {dataset}_{model}: {e}")
            return False

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super(NumpyEncoder, self).default(obj)

    import argparse
    
    MODELS = ['resnet18', 'resnet34', 'dinov2', 'clip']
    DATASETS = ['simple1K', 'Paris_val', 'VOC_val']
    
    parser = argparse.ArgumentParser(description='Generate visualization images')
    parser.add_argument('--dataset', choices=DATASETS, help='Specific dataset to process')
    parser.add_argument('--model', choices=MODELS, help='Specific model to use')
    parser.add_argument('--force', action='store_true', help='Force regeneration of images')
    args = parser.parse_args()
    
    dataset_input = input("Enter dataset (simple1K, Paris_val, VOC_val): ")
    model_input = input("Enter model (resnet18, resnet34, dinov2, clip): ")
    
    if dataset_input and model_input:
        regenerate_images_from_results(dataset_input, model_input, force_regenerate=True)
    elif dataset_input:
        for model in MODELS:
            regenerate_images_from_results(dataset_input, model, force_regenerate=True)
    elif model_input:
        for dataset in DATASETS:
            regenerate_images_from_results(dataset, model_input, force_regenerate=True)
    else:
        for dataset in DATASETS:
            for model in MODELS:
                regenerate_images_from_results(dataset, model, force_regenerate=True)

else:
    print("Non valid option")
