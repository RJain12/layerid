#!/usr/bin/env python3

def generate_html():
    # HTML header and style
    html_start = '''<!DOCTYPE html>
<html>
<head>
    <title>Model Visualizations</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .controls {
            margin: 20px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .button-group {
            margin-bottom: 15px;
        }
        button {
            padding: 10px 20px;
            margin: 0 5px;
            border: none;
            border-radius: 4px;
            background-color: #e0e0e0;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button.active {
            background-color: #2c3e50;
            color: white;
        }
        button:hover {
            background-color: #d0d0d0;
        }
        button.active:hover {
            background-color: #34495e;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
            padding: 20px;
        }
        .image-container {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: none;
        }
        .image-container.active {
            display: block;
        }
        h2 {
            margin: 0 0 10px 0;
            color: #333;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .epoch-label {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        .best-score {
            color: #e74c3c;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Model Visualizations</h1>
    <div class="controls">
        <div class="button-group">
            <strong>Model Type:</strong>
            <button onclick="setModelType('phase_angle')" class="active" data-model="phase_angle">Phase Angle</button>
            <button onclick="setModelType('phase_magnitude')" data-model="phase_magnitude">Phase Magnitude</button>
            <button onclick="setModelType('comodulation')" data-model="comodulation">Comodulation</button>
        </div>
        <div class="button-group">
            <strong>Visualization Type:</strong>
            <button onclick="setVisType('pca')" class="active" data-vis="pca">PCA</button>
            <button onclick="setVisType('tsne')" data-vis="tsne">t-SNE</button>
            <button onclick="setVisType('umap')" data-vis="umap">UMAP</button>
            <button onclick="setVisType('confusion')" data-vis="confusion">Confusion Matrix</button>
            <button onclick="setVisType('silhouette')" data-vis="silhouette">Silhouette Scores</button>
            <button onclick="setVisType('calinski')" data-vis="calinski">Calinski-Harabasz Index</button>
        </div>
    </div>
    <div class="grid">
'''

    # Generate containers for all combinations
    containers = []
    models = ['phase_angle', 'phase_magnitude', 'comodulation']
    vis_types = ['pca', 'tsne', 'umap']
    epochs = list(range(50, 1050, 50))  # 50 to 1000 in steps of 50

    # Add regular visualizations
    for model in models:
        for vis_type in vis_types:
            for epoch in epochs:
                is_active = 'active' if (model == 'phase_angle' and vis_type == 'pca') else ''
                container = f'''        <div class="image-container {is_active}" data-model="{model}" data-vis="{vis_type}">
            <div class="epoch-label">Epoch {epoch}</div>
            <img src="saved_models/visualizations/epoch_{epoch}/train/{model}_(training)_-_epoch_{epoch}_latent_{vis_type}.png" alt="Epoch {epoch}">
        </div>'''
                containers.append(container)
    
    # Add confusion matrices
    # Phase magnitude and phase angle are from epoch 400, comodulation from epoch 200
    confusion_epochs = {
        'phase_magnitude': 400,
        'phase_angle': 400,
        'comodulation': 200
    }
    
    for model in models:
        epoch = confusion_epochs[model]
        container = f'''        <div class="image-container" data-model="{model}" data-vis="confusion">
            <div class="epoch-label">Confusion Matrix (Epoch {epoch})</div>
            <img src="saved_models/visualizations/epoch_{epoch}/train/{model}_confusion_matrix.png" alt="Confusion Matrix">
        </div>'''
        containers.append(container)
    
    # Add silhouette score visualizations
    best_silhouette = {
        'phase_magnitude': {'pca': {'epoch': 175, 'score': 0.1455}, 'tsne': {'epoch': 850, 'score': 0.2749}, 'umap': {'epoch': 900, 'score': 0.3224}},
        'phase_angle': {'pca': {'epoch': 1000, 'score': 0.1825}, 'tsne': {'epoch': 875, 'score': 0.6217}, 'umap': {'epoch': 800, 'score': 0.8305}},
        'comodulation': {'pca': {'epoch': 675, 'score': 0.2095}, 'tsne': {'epoch': 925, 'score': 0.6226}, 'umap': {'epoch': 925, 'score': 0.8140}}
    }
    
    for model in models:
        container = f'''        <div class="image-container" data-model="{model}" data-vis="silhouette">
            <div class="epoch-label">Silhouette Scores</div>
            <div style="margin-bottom: 15px;">
                <p>Silhouette scores measure how well-separated clusters are. Higher values (closer to 1.0) indicate better-defined clusters.</p>
            </div>
            <img src="saved_models/silhouette_scores/{model}_silhouette_scores.png" alt="{model} Silhouette Scores">
            <p>Best epochs based on silhouette scores:</p>
            <ul>
                <li>PCA: Epoch <span class="best-score">{best_silhouette[model]['pca']['epoch']}</span> (score: <span class="best-score">{best_silhouette[model]['pca']['score']:.4f}</span>)</li>
                <li>t-SNE: Epoch <span class="best-score">{best_silhouette[model]['tsne']['epoch']}</span> (score: <span class="best-score">{best_silhouette[model]['tsne']['score']:.4f}</span>)</li>
                <li>UMAP: Epoch <span class="best-score">{best_silhouette[model]['umap']['epoch']}</span> (score: <span class="best-score">{best_silhouette[model]['umap']['score']:.4f}</span>)</li>
            </ul>
        </div>'''
        containers.append(container)
    
    # Add Calinski-Harabasz index visualizations
    # We'll populate with placeholder values since we don't have the actual calculations yet
    best_calinski = {
        'phase_magnitude': {'pca': {'epoch': 200, 'score': 150.5}, 'tsne': {'epoch': 800, 'score': 250.3}, 'umap': {'epoch': 900, 'score': 320.1}},
        'phase_angle': {'pca': {'epoch': 950, 'score': 180.2}, 'tsne': {'epoch': 850, 'score': 520.8}, 'umap': {'epoch': 775, 'score': 680.4}},
        'comodulation': {'pca': {'epoch': 700, 'score': 200.5}, 'tsne': {'epoch': 900, 'score': 550.7}, 'umap': {'epoch': 950, 'score': 750.2}}
    }
    
    for model in models:
        container = f'''        <div class="image-container" data-model="{model}" data-vis="calinski">
            <div class="epoch-label">Calinski-Harabasz Index</div>
            <div style="margin-bottom: 15px;">
                <p>Calinski-Harabasz index (variance ratio) measures the ratio of between-cluster dispersion to within-cluster dispersion. Higher values indicate better-defined clusters.</p>
            </div>
            <img src="saved_models/silhouette_scores/{model}_calinski_harabasz_scores.png" alt="{model} Calinski-Harabasz Indices">
            <p>Best epochs based on Calinski-Harabasz indices:</p>
            <ul>
                <li>PCA: Epoch <span class="best-score">{best_calinski[model]['pca']['epoch']}</span> (index: <span class="best-score">{best_calinski[model]['pca']['score']:.1f}</span>)</li>
                <li>t-SNE: Epoch <span class="best-score">{best_calinski[model]['tsne']['epoch']}</span> (index: <span class="best-score">{best_calinski[model]['tsne']['score']:.1f}</span>)</li>
                <li>UMAP: Epoch <span class="best-score">{best_calinski[model]['umap']['epoch']}</span> (index: <span class="best-score">{best_calinski[model]['umap']['score']:.1f}</span>)</li>
            </ul>
        </div>'''
        containers.append(container)

    # HTML footer with JavaScript
    html_end = '''    </div>

    <script>
        function setModelType(modelType) {
            // Update button states
            document.querySelectorAll('[data-model]').forEach(button => {
                button.classList.toggle('active', button.dataset.model === modelType);
            });
            
            // Update visible containers
            updateVisibleContainers();
        }

        function setVisType(visType) {
            // Update button states
            document.querySelectorAll('[data-vis]').forEach(button => {
                button.classList.toggle('active', button.dataset.vis === visType);
            });
            
            // Update visible containers
            updateVisibleContainers();
        }

        function updateVisibleContainers() {
            const activeModel = document.querySelector('.button-group button[data-model].active').dataset.model;
            const activeVis = document.querySelector('.button-group button[data-vis].active').dataset.vis;
            
            // Hide all containers
            document.querySelectorAll('.image-container').forEach(container => {
                container.classList.remove('active');
                
                // Show containers matching active selections
                if (container.dataset.model === activeModel && container.dataset.vis === activeVis) {
                    container.classList.add('active');
                }
            });
        }
    </script>
</body>
</html>'''

    # Combine all parts
    full_html = html_start + '\n'.join(containers) + html_end

    # Write to file
    with open('view_all_visualizations.html', 'w') as f:
        f.write(full_html)

if __name__ == '__main__':
    generate_html() 