

python -m venv venv
venv\Scripts\activate
python -m ensurepip --upgrade
pip install Flask 
pip install Flask-Login 
pip install Flask-SQLAlchemy 
pip install deepface 
pip install opencv-python-headless 
pip install tensorflow
pip install tf-keras
pip install scipy

pip freeze > requirements.txt

then enter

python app.py 