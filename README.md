# Welcome to WakeUp project !

To get started, please follow these steps :


### Virtual env

In the terminal type : 

- pip install pipenv (install globally pipenv)

- git clone git@github.com:Paul-de-Barbeyrac/wakeup.git

- cd into the created directory

- export PIPENV_VENV_IN_PROJECT=True

- pipenv install

- pipenv shell (to activate virtual env. You should see (wakeup) to the left of your terminal)

You are good to go, just run : python main.py

If you need to install other libraries, just run **pipenv install library_name** instead of pip install library_name.

All dependencies requirements are listed in the Pipfile and they will be installed inside a .venv folder in your project directory.


### Dataset

- cd into the root of the project directory and paste the following terminal commands
- curl http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip --output data.zip
- unzip data.zip
- mv mrlEyes_2018_01 dataset
- rm dataset/stats_2018_01.ods
- rm data.zip

### How to launch webserver with Django

- In your terminal (only needed once) : python manage.py migrate (it will create a sqlite database necessary for Django with some tables)
- Everytime you want to launch the local webrowser : python manage.py runserver
- Ctrl + C to stop webserver


You are setup, let's work !




