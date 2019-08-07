# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# Running the API

Build the image using the following command

```bash
$ docker build -t flower-image-classifier:latest .
```

Run the Docker container using the command shown below.

```bash
$  docker run -p 80:80 flower-image-classifier
```

The application will be accessible at http://localhost or if you are using boot2docker then first find ip address using `$ boot2docker ip` and the use the ip `http://<host_ip>`

# TODOs

- Docker container hangs when executing `model.forward`. PyTorch (CPU) issue?
- HTTPS support
