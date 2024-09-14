from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from cnn_model import train_cnn_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'



# Page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

# Gestion du téléchargement des images et des paramètres
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('images')
        learning_rate = float(request.form.get('learning_rate'))
        epochs = int(request.form.get('epochs'))

        for file in files:
            if file:
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        # Entrainement du modèle avec les images téléchargées
        accuracy = train_cnn_model(app.config['UPLOAD_FOLDER'], learning_rate, epochs)

        return render_template('result.html', accuracy=accuracy)

# Servir les fichiers statiques (comme les images uploadées)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    if request.method == 'POST':
        files = request.files.getlist('images')
        learning_rate = float(request.form.get('learning_rate'))
        epochs = int(request.form.get('epochs'))

        # Vérifier si des fichiers ont été uploadés
        if not files or len(files) == 0:
            return "Aucune image n'a été téléchargée."

        # Enregistrer les fichiers dans le dossier
        for file in files:
            if file:
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        # Vérification supplémentaire si le dossier contient bien des images
        if not os.listdir(app.config['UPLOAD_FOLDER']):
            return "Aucune image n'a été trouvée dans le dossier après l'upload."

        # Entrainement du modèle avec les images téléchargées
        accuracy = train_cnn_model(app.config['UPLOAD_FOLDER'], learning_rate, epochs)

        return render_template('result.html', accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
