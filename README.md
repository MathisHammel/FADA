# FADA

[Access it here](https://fada.h25.io)

This tool/PoC is designed to reliably detect pictures coming from [ThisPersonDoesNotExist](https://thispersondoesnotexist.com) without using any heuristic techniques.

It's based on the fact that TPDNE has a global cache and serves the same image to everyone on the planet (refreshed every ~1.2 seconds). We can download all images and index them, in order to search the database.

FADA is an acronym which means something super smart but I forgot it...

You will need an instance of Elasticsearch Open Distro running on localhost, needed for indexing and searching k-NN embedding vectors.

## How it works

There are two scripts which need to run:

- downloader.py : this one has to keep running permanently. It runs three concurrent processes in an infinite loop which save pictures from ThisPersonDoesNotExist in the download/ folder.
- server.py : it serves two purposes. First, this script opens a Flask server for the UI (opened on 0.0.0.0 by default). It also runs an infinite loop which consumes pictures in download/ and stores them in the Elasticsearch index.

## Usage

`./start.sh` launches everything (and kills potentially existing instances of FADA to avoid conflicts)

This command needs root privileges since it listens on port 80/443, but you can get rid of that by changing the port used by Flask.

## Note

As usual, all PRs are welcome, but I do not intend to keep fada.h25.io online permanently since it's more of a PoC. If you are interested in hosting it, get in touch!