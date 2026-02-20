

make setup-miniwob
this git clones miniwob

it sets .env which is not correct. It sets to a file under html

cd to miniwob-plusplus/miniwob/html and start a python server
python -m http.server 8000

then in another terminal window set env MINIWOB_URL=http://127.0.0.1/miniwob
and run your python agent files from another directory, not the html dir above to reduce confusion

the python agent will use the python server to get the test cases

verify the python server works for your agent 

check with 2 requests
curl -I http://127.0.0.1:8000/miniwob/core.js
curl -I http://127.0.0.1:8000/miniwob/click-button.html

click-button.html uses d3 and jquery. it is dynamic and requires core.js to work. 
make sure to verify core.js works statically before going to agent code. Agent code in colab uses core.js and there is a race condition in colab which can cause core.js to not work when loading click-button.html. 
Should check it see if DOM ready signal. 



