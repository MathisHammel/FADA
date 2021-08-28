sudo pkill -f "server.py"
sudo pkill -f "downloader.py"
sudo nohup python -u downloader.py 2>&1 > downloader.log &
sudo -E nohup python -u server.py 2>&1 > server.log &
echo "OK"
tail -f downloader.log server.log
