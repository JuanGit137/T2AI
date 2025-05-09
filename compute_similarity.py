import numpy as np
import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import euclidean_distances

# Corregimos las rutas para tu configuración local
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'T2images', 'simple1K')
image_dir = os.path.join(data_dir, 'images')
list_of_images = os.path.join(data_dir, 'list_of_images.txt')

# Usamos el nombre correcto de dataset (distingue mayúsculas/minúsculas)
DATASET = 'simple1K'  # Asegúrate que coincida exactamente con el nombre usado en compute_features.py
MODEL = 'resnet34'
feat_file = os.path.join('data', f'feat_{MODEL}_{DATASET}.npy')
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados')

# Crear carpeta de resultados si no existe
os.makedirs(output_dir, exist_ok=True)

if __name__ == '__main__':
    print(f"Buscando archivo de características en: {feat_file}")
    print(f"Buscando lista de imágenes en: {list_of_images}")
    
    try:
        # Cargar lista de imágenes
        with open(list_of_images, "r") as file: 
            files = [f.strip().split('\t') for f in file]
        print(f"Se encontraron {len(files)} imágenes en la lista")
            
        # Cargar características
        feats = np.load(feat_file)    
        print(f"Características cargadas: {feats.shape}")
        
        # Calcular distancias euclidianas entre todas las características
        dist = euclidean_distances(feats)
        
        # Obtener índices ordenados por distancia
        dist_idx = np.argsort(dist, axis=1)
        
        # Número de imágenes similares a mostrar (sin contar la consulta)
        k = 10
        
        # Realizar 10 iteraciones con consultas aleatorias
        iteration_results = []
        
        for iteration in range(10):
            # Seleccionar una imagen aleatoria como consulta
            query = np.random.permutation(dist.shape[0])[0]
            best_idx = dist_idx[query, :k+1]  # Incluye la propia imagen de consulta
            
            # Obtener la clase de la imagen de consulta (si está disponible)
            query_class = files[query][1] if len(files[query]) > 1 else "desconocida"
            
            # Contar cuántas imágenes coinciden con la clase de la consulta
            matching_classes = 0
            for idx in best_idx:
                if len(files[idx]) > 1 and files[idx][1] == query_class:
                    matching_classes += 1
            
            # Guardar resultados de esta iteración
            result = {
                'iteration': iteration,
                'query_idx': query,
                'query_class': query_class,
                'best_idx': best_idx,
                'matching_classes': matching_classes
            }
            iteration_results.append(result)
            
            print(f"\nIteración {iteration+1}/10:")
            print(f"Imagen de consulta: {files[query][0]} - Clase: {query_class}")
            print(f"Coincidencias de clase: {matching_classes-1}/{k}")
        
        # Ordenar iteraciones por número de coincidencias (mejores primero)
        sorted_iterations = sorted(iteration_results, key=lambda x: x['matching_classes'], reverse=True)
        
        # Seleccionar las 5 mejores y 5 peores iteraciones
        best_iterations = sorted_iterations[:5]
        worst_iterations = sorted_iterations[-5:]
        
        # Función para visualizar resultados de una iteración
        def visualize_iteration(iteration_data, title, filename):
            query_idx = iteration_data['query_idx']
            best_idx = iteration_data['best_idx']
            query_class = iteration_data['query_class']
            
            fig, ax = plt.subplots(1, k+1, figsize=(15, 3))
            fig.suptitle(f"{title} - Clase de consulta: {query_class} - Coincidencias: {iteration_data['matching_classes']-1}/{k}")
            
            # Mostrar imágenes
            for i, idx in enumerate(best_idx):
                try:
                    img_filename = os.path.join(image_dir, files[idx][0])
                    im = io.imread(img_filename)
                    im = transform.resize(im, (64, 64))
                    ax[i].imshow(im)
                    ax[i].set_axis_off()
                    
                    # Mostrar clase y si coincide con la consulta
                    if len(files[idx]) > 1:
                        img_class = files[idx][1]
                        match = "✓" if img_class == query_class else "✗"
                        ax[i].set_title(f"{img_class} {match}")
                except Exception as e:
                    print(f"Error al cargar imagen {files[idx][0]}: {e}")
                    ax[i].text(0.5, 0.5, "Error", ha='center', va='center')
                    ax[i].set_axis_off()
            
            # Destacar la imagen de consulta
            ax[0].patch.set(lw=6, ec='b')
            ax[0].set_axis_on()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()
            
        # Visualizar las mejores 5 iteraciones
        for i, iteration_data in enumerate(best_iterations):
            title = f"TOP {i+1}: Mejor clasificación"
            filename = f"mejor_clasificacion_{i+1}.png"
            visualize_iteration(iteration_data, title, filename)
        
        # Visualizar las peores 5 iteraciones
        for i, iteration_data in enumerate(worst_iterations):
            title = f"BOTTOM {i+1}: Peor clasificación"
            filename = f"peor_clasificacion_{i+1}.png"
            visualize_iteration(iteration_data, title, filename)
        
        # Crear una visualización resumen con mejores y peores resultados
        fig_best = plt.figure(figsize=(15, 12))
        fig_best.suptitle("Las 5 MEJORES Clasificaciones", fontsize=16)
        
        for i, iteration_data in enumerate(best_iterations):
            query_idx = iteration_data['query_idx']
            best_idx = iteration_data['best_idx'][:5]  # Solo mostrar 5 imágenes por resultado
            query_class = iteration_data['query_class']
            
            for j, idx in enumerate(best_idx):
                ax = fig_best.add_subplot(5, 5, i*5 + j + 1)
                try:
                    img_filename = os.path.join(image_dir, files[idx][0])
                    im = io.imread(img_filename)
                    im = transform.resize(im, (64, 64))
                    ax.imshow(im)
                    ax.set_axis_off()
                    
                    if j == 0:  # Imagen de consulta
                        ax.set_title(f"Query: {query_class}", fontsize=10)
                        ax.patch.set(lw=3, ec='b')
                    elif len(files[idx]) > 1:
                        img_class = files[idx][1]
                        match = "✓" if img_class == query_class else "✗"
                        ax.set_title(f"{img_class} {match}", fontsize=8)
                except Exception as e:
                    ax.text(0.5, 0.5, "Error", ha='center', va='center')
                    ax.set_axis_off()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, "resumen_mejores_clasificaciones.png"))
        plt.close()
        
        # Crear visualización para peores resultados
        fig_worst = plt.figure(figsize=(15, 12))
        fig_worst.suptitle("Las 5 PEORES Clasificaciones", fontsize=16)
        
        for i, iteration_data in enumerate(worst_iterations):
            query_idx = iteration_data['query_idx']
            best_idx = iteration_data['best_idx'][:5]  # Solo mostrar 5 imágenes por resultado
            query_class = iteration_data['query_class']
            
            for j, idx in enumerate(best_idx):
                ax = fig_worst.add_subplot(5, 5, i*5 + j + 1)
                try:
                    img_filename = os.path.join(image_dir, files[idx][0])
                    im = io.imread(img_filename)
                    im = transform.resize(im, (64, 64))
                    ax.imshow(im)
                    ax.set_axis_off()
                    
                    if j == 0:  # Imagen de consulta
                        ax.set_title(f"Query: {query_class}", fontsize=10)
                        ax.patch.set(lw=3, ec='b')
                    elif len(files[idx]) > 1:
                        img_class = files[idx][1]
                        match = "✓" if img_class == query_class else "✗"
                        ax.set_title(f"{img_class} {match}", fontsize=8)
                except Exception as e:
                    ax.text(0.5, 0.5, "Error", ha='center', va='center')
                    ax.set_axis_off()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, "resumen_peores_clasificaciones.png"))
        plt.close()
        
        print("\nProceso completado!")
        print(f"Se han guardado las imágenes de resultados en: {output_dir}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Verifica que los archivos necesarios existan:")
        print(f"- Archivo de características: {feat_file}")
        print(f"- Lista de imágenes: {list_of_images}")
    except Exception as e:
        print(f"Error inesperado: {e}")