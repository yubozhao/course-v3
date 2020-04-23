---
title: "Deploying on Amazon SageMaker with BentoML"
sidebar: home_sidebar
---


# Kubernetes Deployment

In this guide, you will deploy the pet classification model from lesson one as an API
server to Kubernetes using BentoML.

## Setup

1. A Kubernetes enabled cluster or machine.
    * learn more about installation: https://kubernetes.io/docs/setup/
    * This guide uses Kubernetes' recommend learning environment `minikube`.
    `minikube` installation: https://kubernetes.io/docs/setup/learning-environment/minikube/
    * `kubectl` CLI tool: https://kubernetes.io/docs/tasks/tools/install-kubectl/
2. Docker and Docker Hub is properly installed and configured on your local system
    * Installation instruction: https://www.docker.com/get-started
3. Python (3.6 or above) and required packages: `bentoml`, `fastai`
    * ```pip install bentoml fastai```


## Save pet classification model with BentoML

Follow the notebook from Fastai lesson one to train a pet classification model.

An example base on that notebook is the following:

```python
from fastai.vision import *

path = untar_data(URLs.PETS)
path_img = path/'images'
fnames = get_image_files(path_img)
pat = re.compile(r'/([^/]+)_\d+.jpg$')
bs=64
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),
                                   size=299, bs=bs//2).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet50, metrics=error_rate)
learn.fit_one_cycle(8)
learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))
```

### Define API server with BentoML

Save the following code to a file named `pet_classification.py`:

```python
from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import FastaiModelArtifact
from bentoml.handlers import FastaiImageHandler

@artifacts([FastaiModelArtifact('pet_classifier')])
@env((auto_pip_dependencies=True)
class PetClassification(BentoService):

    @api(FastaiImageHandler)
    def predict(self, image):
        result = self.artifacts.pet_classifier.predict(image)
        return str(result)
```

This code defines a prediction server using `Fastai` model, asks BentoML to figure out
the required PyPi packages automatically. It also defined an API called `predict`, that
is the entry point to access this prediction service. The API is expecting a `Fastai`
`ImageData` object as its input data.

Run the following code to create a BentoService SavedBundle with the pet classification
model. It's a versioned file archive ready for production deployment.  The archive
contains the prediction service defined above, python code dependencies and PyPi
dependencies, and the trained pet classification model. :

```python
# 1) import the custom BentoService defined above
from pet_classification import PetClassification

# 2) `pack` it with required artifacts
service = PetClassification()
service.pack('pet_classifier', learn)

# 3) save your BentoService
service.save()
```

### Validate the saved bundle by using prediction request with sample data

BentoML automatically process the incoming data into required data format defined in the
API. For the pet classifier BentoService defined above, incoming data will transform to
fastai `ImageData` object.

Use BentoML CLI tool to

```bash
# Replace PATH_TO_TEST_IMAGE_FILE with one of the image from {path_img}
# An example path: /Users/user_name/.fastai/data/oxford-iiit-pet/images/shiba_inu_122.jpg

bentoml run PetClassification:latest predict --input=PATH_TO_TEST_IMAGE_FILE
```

## Deploy to Kubernetes

### Build and push API server image

BentoService SaveBundle directory structured as a docker context, that can be used to
build an API server docker image.

Replace `docker_username` with Docker Hub username:

```bash
saved_path=$(bentoml get IrisClassifier:latest -q | jq -r ".uri.uri")

docker build -t {docker_username}/pet-classifier .
docker push
```

### Apply deployment to Kubernetes

The following code creates a kubernetes resources spec. Replace `{docker_username}`
with Docker Hub username and save to a file called
`pet-classifier.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  labels:
    app: pet-classifier
  name: pet-classifier
spec:
  ports:
  - name: predict
    port: 5000
    targetPort: 5000
  selector:
    app: pet-classifier
  type: LoadBalancer
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: pet-classifier
  name: pet-classifier
spec:
  selector:
    matchLabels:
      app: pet-classifier
  template:
    metadata:
      labels:
        app: pet-classifier
    spec:
      containers:
      - image: {docker_username}/pet-classifier
        name: pet-classifier
        ports:
        - containerPort: 5000
```

Use `kubectl` to apply spec to the kubernetes cluster.

```bash
kubectl apply -f pet-classifier.yaml
```

Check deployment status with `kubectl`:

```bash
kubectl get svc pet-classifier

### Send prediction request

```bash
# Replace PATH_TO_TEST_IMAGE_FILE

curl -i \
    --request POST \
    --header "Content-Type: multipart/form-data" \
    -F "image=@PATH_TOTEST_IMAGE_FILE" \
    localhost:5000/predict
```

### Remote deployment

```bash
kubectl delete -f pet-classifier.yaml
```
