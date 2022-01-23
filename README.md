Stylegan3 latent space explorer web UI
======================================

This project is a simple web UI allowing you to explore the latent space of a stylegan3 generator network.

Requirements
------------

A GPU! Running on CPU is possible (just change 'cuda' to 'cpu' in the sources) but will be horribly slow.

A stylegan3 checkpoint file, copied or symlinked to `stylegan3.pkl`.

python3 and python libraries  waitress, torch, numpy, PIL


Running
-------

- Clone the repository and `git submodule update --init`.
- Run `python3 stylegan-server.py`

It will listen on port 8001 on all interfaces (edit code at the end to change that).

Then open a browser to http://localhost:8001


Features
--------

  - Configureable image size and count in a grid layout
  - Fully random latent vector generation, or
  - Latent vector generation around a given point (image) with a configureable radius


![Screenshot](screenshots/stylegan3-latent-explorer.jpg?raw=true "Screenshot")
