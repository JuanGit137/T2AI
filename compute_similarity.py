import numpy as np
import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# Definición de rutas principales
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'T2images', 'simple1K')
image_dir = os.path.join(data_dir, 'images')
list_of_images = os.path.join(data_dir, 'list_of_images.txt')

DATASET = 'Paris_val'
MODEL = 'resnet34'
feat_file = os.path.join('data', f'feat_{MODEL}_{DATASET}.npy')

# Estructura de carpetas para resultados
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
img_results_dir = os.path.join(results_dir, 'imgresults')
gen_results_dir = os.path.join(results_dir, 'genresults')

# Carpeta específica para las imágenes de este modelo y dataset
output_dir = os.path.join(img_results_dir, f"{DATASET}_{MODEL}")

# Crear estructura de carpetas
os.makedirs(results_dir, exist_ok=True)
os.makedirs(img_results_dir, exist_ok=True)
os.makedirs(gen_results_dir, exist_ok=True)

# Archivo para guardar el diccionario de resultados (ahora en JSON)
results_dict_file = os.path.join(gen_results_dir, 'evaluation_results.json')

# Verificar si ya existe la carpeta específica con contenido
if os.path.exists(output_dir) and os.listdir(output_dir):
    print(f"El directorio {output_dir} ya existe y contiene archivos.")
    print("Los resultados de imágenes no serán sobrescritos.")
    # Crear un directorio alternativo con timestamp para evitar sobrescribir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(img_results_dir, f"{DATASET}_{MODEL}_{timestamp}")
    print(f"Usando directorio alternativo: {output_dir}")

# Crear carpeta para las imágenes específicas de este modelo y dataset
os.makedirs(output_dir, exist_ok=True)

# Clase para ayudar a serializar objetos numpy a JSON
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
    """Cargar el diccionario de resultados o crear uno nuevo si no existe"""
    if os.path.exists(results_dict_file):
        with open(results_dict_file, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("Error al leer el archivo JSON. Creando un nuevo diccionario.")
                return {}
    return {}

def save_results_dict(results_dict):
    """Guardar el diccionario de resultados en JSON"""
    with open(results_dict_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, cls=NumpyEncoder)
    print(f"Diccionario de resultados guardado en {results_dict_file}")

if __name__ == '__main__':
    print(f"Buscando archivo de características en: {feat_file}")
    print(f"Buscando lista de imágenes en: {list_of_images}")
    
    try:
        with open(list_of_images, "r") as file: 
            files = [f.strip().split('\t') for f in file]
        print(f"Se encontraron {len(files)} imágenes en la lista")
            
        feats = np.load(feat_file)    
        print(f"Características cargadas: {feats.shape}")
        
        # Cargar diccionario de resultados
        results_dict = load_results_dict()
        key = f"{DATASET}_{MODEL}"
        
        # Verificar si ya existen resultados en el diccionario
        if key in results_dict:
            print(f"Ya existen resultados para {DATASET} con modelo {MODEL} en el diccionario.")
            print(f"Los resultados del diccionario no serán sobrescritos.")
            print(f"Fecha de evaluación anterior: {results_dict[key].get('timestamp', 'Desconocida')}")
        
        # Calcular similitud coseno
        similarity = cosine_similarity(feats)
        
        # Convertir similitud a distancia
        dist = 1 - similarity
        
        # Ordenar índices por distancia
        dist_idx = np.argsort(dist, axis=1)
        
        k = 10
        
        iteration_results = []
        
        # Realizar 10 iteraciones con consultas aleatorias
        for iteration in range(10):
            query = np.random.permutation(dist.shape[0])[0]
            best_idx = dist_idx[query, :k+1]
            
            query_class = files[query][1] if len(files[query]) > 1 else "desconocida"
            
            matching_classes = 0
            for idx in best_idx:
                if len(files[idx]) > 1 and files[idx][1] == query_class:
                    matching_classes += 1
            
            result = {
                'iteration': iteration,
                'query_idx': int(query),  # Convertir a int para asegurar serialización JSON
                'query_class': query_class,
                'best_idx': best_idx.tolist() if isinstance(best_idx, np.ndarray) else best_idx,
                'matching_classes': int(matching_classes-1)
            }
            iteration_results.append(result)
            
            print(f"\nIteración {iteration+1}/10:")
            print(f"Imagen de consulta: {files[query][0]} - Clase: {query_class}")
            print(f"Coincidencias de clase: {matching_classes-1}/{k}")
        
        # Ordenar iteraciones por número de coincidencias
        sorted_iterations = sorted(iteration_results, key=lambda x: x['matching_classes'], reverse=True)
        
        # Seleccionar las 5 mejores y 5 peores iteraciones
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
                except Exception as e:
                    print(f"Error al cargar imagen {files[idx][0]}: {e}")
                    ax[i].text(0.5, 0.5, "Error", ha='center', va='center')
                    ax[i].set_axis_off()
            
            ax[0].patch.set(lw=6, ec='b')
            ax[0].set_axis_on()
            
            plt.tight_layout()
            img_path = os.path.join(output_dir, filename)
            plt.savefig(img_path)
            plt.close()
            return img_path
        
        # Visualizar y guardar las mejores clasificaciones
        best_image_paths = []
        for i, iteration_data in enumerate(best_iterations):
            title = f"TOP {i+1}: Best classification"
            filename = f"mejor_clasificacion_{i+1}.png"
            img_path = visualize_iteration(iteration_data, title, filename)
            best_image_paths.append(img_path)
        
        # Visualizar y guardar las peores clasificaciones
        worst_image_paths = []
        for i, iteration_data in enumerate(worst_iterations):
            title = f"BOTTOM {i+1}: Peor clasificación"
            filename = f"peor_clasificacion_{i+1}.png"
            img_path = visualize_iteration(iteration_data, title, filename)
            worst_image_paths.append(img_path)
        
        # Guardar resultados en el diccionario si no existen previamente
        if key not in results_dict:
            # Calcular mAP aproximado
            total_matches = 0
            total_queries = 0
            for result in iteration_results:
                matches = result['matching_classes']
                total_matches += matches
                total_queries += 1
                
            mAP_approx = total_matches / (total_queries * k) if total_queries > 0 else 0
            
            # Guardar los resultados en el diccionario
            results_dict[key] = {
                'dataset': DATASET,
                'model': MODEL,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'mAP': float(mAP_approx),  # Convertir a float para asegurar serialización JSON
                'best_iterations': best_iterations,
                'worst_iterations': worst_iterations,
                'best_image_paths': best_image_paths,
                'worst_image_paths': worst_image_paths,
                'output_dir': output_dir
            }
            
            # Guardar el diccionario actualizado
            save_results_dict(results_dict)
            print(f"Resultados guardados en el diccionario para {key}")
        
        print("\nProceso completado!")
        print(f"Se han guardado las imágenes de resultados en: {output_dir}")
        print(f"Los resultados generales se encuentran en: {gen_results_dir}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Verifica que los archivos necesarios existan:")
        print(f"- Archivo de características: {feat_file}")
        print(f"- Lista de imágenes: {list_of_images}")
    except Exception as e:
        print(f"Error inesperado: {e}")