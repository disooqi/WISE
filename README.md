<!-- https://guides.github.com/features/mastering-markdown/ -->
![Image of Yaktocat](https://github.com/CoDS-GCS/WISE/blob/dev/logo.png)

* WISE has been developed using Python 3.7
* Make sure you create a virtual environment, activated it and the thin pip install the requirement.txt file comes with the project. 
* Make sure the file `wiki-news-300d-1M.txt` is under `word_embedding` directory
* Start the Word Embedding server using the command `python word_embedding/server.py 127.0.0.1 9600`.

WISE Project is work under development; expect some abnormality during installing and using the software.

License: Not determined yet!

Documentation: https://wise.eldesouki.ca

Usage
-----
First you need to import wise 

``from wise import Wise``

and then create an instance as following:

``wisely = Wise()``

and then
 
 ``answers = wisely.ask("Who was the doctoral supervisor of Albert Einstein?")``
