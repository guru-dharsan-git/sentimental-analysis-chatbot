 cd chatbot-deployment
python3 -m venv venv
. venv/bin/activate
Install dependencies
$ (venv) pip install Flask torch torchvision nltk
Install nltk package
$ (venv) python
>>> import nltk
>>> nltk.download('punkt')
