<p align="center">
  <img width="460" height="200" src="https://user-images.githubusercontent.com/1148046/79498304-14c00680-7ff7-11ea-9b68-a4749138ce25.png">
</p>

* WISE has been developed using Python 3.7
* Make sure you create a virtual environment, activated it and the thin pip install the requirement.txt file comes with the project. 
* Make sure the file `wiki-news-300d-1M.txt` is under `word_embedding` directory
* Start the Word Embedding server using the command `python word_embedding/server.py 127.0.0.1 9600`.

WISE Project is work under development; expect some abnormality during installing and using the software.

Evaluation
===========
To evaluate the system on QALD dataset.

QALD 6 or Later
------------------
`sudo apt-get remove --purge ruby-full`

`sudo apt-get update`

`sudo apt-get install git-core curl zlib1g-dev build-essential libssl-dev libreadline-dev libyaml-dev libsqlite3-dev sqlite3 
libxml2-dev libxslt1-dev libcurl4-openssl-dev software-properties-common libffi-dev`

`cd`

`git clone https://github.com/rbenv/rbenv.git ~/.rbenv`

`echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.bashrc`

`echo 'eval "$(rbenv init -)"' >> ~/.bashrc`

`exec $SHELL`

`git clone https://github.com/rbenv/ruby-build.git ~/.rbenv/plugins/ruby-build`

`echo 'export PATH="$HOME/.rbenv/plugins/ruby-build/bin:$PATH"' >> ~/.bashrc`

`exec $SHELL`

`rbenv install 2.7.0`

`rbenv global 2.7.0`

`gem install bundler`

`rbenv rehash`

`gem install nokogiri mustache multiset`

`git clone https://github.com/ag-sc/QALD.git`

Rename `QALD/6/data/qald-6-test-multilingual.json` into `QALD/6/data/dbpedia-test.json`

`cd projects/QALD/6/scripts/`

`ruby evaluation.rb wise/output/WISE_result_20200325-163649.json`
