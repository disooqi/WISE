<p align="center">
  <img width="460" height="200" src="https://user-images.githubusercontent.com/1148046/79498304-14c00680-7ff7-11ea-9b68-a4749138ce25.png">
</p>

* WISE has been developed using Python 3.7
* Make sure you create a virtual environment, activated it and the thin pip install the requirement.txt file comes with the project. 
* Make sure the file `wiki-news-300d-1M.txt` is under `word_embedding` directory
* Start the Word Embedding server using the command `python word_embedding/server.py 127.0.0.1 9600`.

WISE Project is work under development; expect some abnormality during installing and using the software.

License: Not determined yet!

Documentation: http://cods.encs.concordia.ca/wise

Usage
-----
First you need to import wise 

``from wise import Wise``

and then create an instance as following:

``wisely = Wise()``

and then
 
 ``answers = wisely.ask("Who was the doctoral supervisor of Albert Einstein?")``
 
 Install WISE
 ------------
 
1) Install Python 3.7

2) Clone WISE into some directory. Lets call it <wise_home> 

``git clone https://github.com/CoDS-GCS/WISE.git wise``

3) copy file wiki-news-300d-1M.txt into directory <wise_home>/word_embedding/ 

4) check out branch that uses the new version of AllenNLP

``git checkout allennlp-1.0.0``

5) Installing pipenv 

``pip install --user pipenv
echo 'export PIPENV_VENV_IN_PROJECT=1' >> ~/.bashrc``

You might need to restart the system.

6) change directory into <wise_home>, and run the following:

``pipenv install``

7) Run the word embedding server

``pipenv run python word_embedding/server.py 127.0.0.1 9600``

8) To run evaluation over QALD 9, change directory into "<wise_home>/evaluation", and then run the following into in the shell:

``export PYTHONPATH="<wise_home>"``

``pipenv run  python qald9eval.py``
