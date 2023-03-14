# COMP9517 Project

- Developed a real-time video tracking system using DeepSORT and YOLOv5 to accurately detect and track pedestrians, achieving a precision of 88.5% and a recall of 68.5%.
- Implemented algorithms to analyze pedestrian behaviour over time, including counting the number of pedestrians walking in groups and alone, and tracking group formation and destruction.
- Demonstrated the effectiveness of the tracking system through a MOTA score of 49.5%.

## Installation

### Setup the environment

1. `cd group_project`
2. `pip install -r requirements.txt` (You may also choose to install all packages on your own)

### Data preprocessing of the original dataset

1. You may run `python get_labels.py` to convert pixel-label maps to bounding boxes
2. You may run `python images_to_movie.py` to convert images from dataset to videos
(We already provide the videos generated from images in official dataset and bounding box labels)

### Run the code for each tasks and evaluation

1. Task 1: `python task1.py`
2. Task 2: `python task2.py`
3. Task 3: `python task3.py`
4. Evaluation: `python evaluation.py`

### The directory structure should be the same as below

```tree
.
├── README.md
├── data
│   ├── hyps
│   │   ├── hyp.finetune.yaml
│   │   ├── hyp.finetune_objects365.yaml
│   │   ├── hyp.scratch-high.yaml
│   │   ├── hyp.scratch-low.yaml
│   │   ├── hyp.scratch-med.yaml
│   │   └── hyp.scratch.yaml
│   ├── prepare_train_data
│   │   ├── get_labels.py
│   │   ├── ground_truth.txt
│   │   ├── maps_to_bbox.py
│   │   └── predit.txt
│   ├── video
│   │   ├── images_to_movie.py
│   │   ├── test_01.mp4
│   │   └── test_07.mp4
│   └── yolo.yaml
├── deep_sort
│   ├── configs
│   │   └── deep_sort.yaml
│   ├── deep_sort
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── deep_sort.cpython-38.pyc
│   │   ├── deep
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── feature_extractor.cpython-38.pyc
│   │   │   │   └── model.cpython-38.pyc
│   │   │   ├── checkpoint
│   │   │   │   └── ckpt.t7
│   │   │   ├── evaluate.py
│   │   │   ├── feature_extractor.py
│   │   │   ├── model.py
│   │   │   ├── original_model.py
│   │   │   ├── prepare_car.py
│   │   │   ├── prepare_person.py
│   │   │   ├── test.py
│   │   │   └── train.py
│   │   ├── deep_sort.py
│   │   └── sort
│   └── utils
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-38.pyc
│       │   └── parser.cpython-38.pyc
│       ├── asserts.py
│       ├── draw.py
│       ├── evaluation.py
│       ├── io.py
│       ├── json_logger.py
│       ├── log.py
│       ├── parser.py
│       └── tools.py
├── detect.py
├── easydict
│   ├── CHANGES
│   ├── LICENSE
│   ├── MANIFEST.in
│   ├── README.rst
│   ├── easydict
│   │   └── __init__.py
│   └── setup.py
├── evaluate.py
├── models
│   ├── common.py
│   ├── experimental.py
│   ├── hub
│   │   ├── anchors.yaml
│   │   ├── yolov3-spp.yaml
│   │   ├── yolov3-tiny.yaml
│   │   ├── yolov3.yaml
│   │   ├── yolov5-bifpn.yaml
│   │   ├── yolov5-fpn.yaml
│   │   ├── yolov5-p2.yaml
│   │   ├── yolov5-p34.yaml
│   │   ├── yolov5-p6.yaml
│   │   ├── yolov5-p7.yaml
│   │   ├── yolov5-panet.yaml
│   │   ├── yolov5l6.yaml
│   │   ├── yolov5m6.yaml
│   │   ├── yolov5n6.yaml
│   │   ├── yolov5s-ghost.yaml
│   │   ├── yolov5s-transformer.yaml
│   │   ├── yolov5s6.yaml
│   │   └── yolov5x6.yaml
│   ├── tf.py
│   ├── yolo.py
│   ├── yolov5l.yaml
│   ├── yolov5m.yaml
│   ├── yolov5n.yaml
│   ├── yolov5s-dot.yaml
│   ├── yolov5s.yaml
│   └── yolov5x.yaml
├── objdetector.py
├── objtracker.py
├── runs
│   ├── detect
│   │   └── exp2
│   │       └── labels
│   └── train
│       └── exp
│           ├── F1_curve.png
│           ├── PR_curve.png
│           ├── P_curve.png
│           ├── R_curve.png
│           ├── confusion_matrix.png
│           ├── events.out.tfevents.1658296352.son.41944.0
│           ├── hyp.yaml
│           ├── opt.yaml
│           ├── results.csv
│           ├── results.png
│           ├── train_batch0.jpg
│           ├── train_batch1.jpg
│           ├── train_batch2.jpg
│           ├── val_batch0_labels.jpg
│           ├── val_batch0_pred.jpg
│           ├── val_batch1_labels.jpg
│           ├── val_batch1_pred.jpg
│           ├── val_batch2_labels.jpg
│           ├── val_batch2_pred.jpg
│           └── weights
│               ├── best.pt
│               └── last.pt
├── task1.py
├── task2.py
├── task3.py
├── train.py
├── utils
│   ├── activations.py
│   ├── augmentations.py
│   ├── autoanchor.py
│   ├── autobatch.py
│   ├── aws
│   │   ├── __init__.py
│   │   ├── mime.sh
│   │   ├── resume.py
│   │   └── userdata.sh
│   ├── callbacks.py
│   ├── datasets.py
│   ├── downloads.py
│   ├── flask_rest_api
│   │   ├── README.md
│   │   ├── example_request.py
│   │   └── restapi.py
│   ├── general.py
│   ├── google_app_engine
│   │   ├── Dockerfile
│   │   ├── additional_requirements.txt
│   │   └── app.yaml
│   ├── loggers
│   │   ├── __init__.py
│   │   └── wandb
│   │       ├── README.md
│   │       ├── __init__.py
│   │       ├── log_dataset.py
│   │       ├── sweep.py
│   │       ├── sweep.yaml
│   │       └── wandb_utils.py
│   ├── loss.py
│   ├── metrics.py
│   ├── plots.py
│   ├── stereo.py
│   └── torch_utils.py
├── val.py
└── weights
    ├── best.pt
    └── yolov5s.pt
```
