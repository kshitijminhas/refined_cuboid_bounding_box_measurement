# refined_cuboid_bounding_box_measurement
Rutgers ECE 332:561 (Machine Vision)
Fall 2020
Faith Johnson, Kshitij Minhas

1. git clone https://github.com/kshitijminhas/refined_cuboid_bounding_box_measurement
2. mkdir third_party
3. mkdir third_party/weights
4. cd third_party
5. git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
6. git clone https://github.com/kshitijminhas/DenseDepth
7. cd DenseDepth
8. git checkout mv_project
9. Download Densedepth model to weights folder, and edit --densedepth_model arg in refined_cuboid..../main.py
10. Download and place Yolo weights to folder: ..third_party/PyTorch-YOLOv3/weights/