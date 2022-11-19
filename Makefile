start: stop
	python3 downloader.py > logs/downloader.$(shell date +%s).log 2>&1  &
	python3 server.py > logs/server.$(shell date +%s).log 2>&1 &

stop:
	-pkill -f "server.py"
	-pkill -f "downloader.py"

watch:
	tail -f logs/*.log

build:
	python3 -m venv env
	. env/bin/activate
	python3 -m pip install -r requirements.txt