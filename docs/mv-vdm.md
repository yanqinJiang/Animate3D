# Training
Use the following command to train:
```bash
bash train.sh ${machine_num} ${gpu_per_machine} ${config_name} 
```
For example:
```bash
bash train.sh 1 2 train 
```
You can find the outputs in `outputs/vdm/train` folder. 
# Inference
Use the following command to inference:
```bash
bash inference.sh ${gpu_id} ${config_file} ${prompt} ${ip_image_root} ${ip_image_name} ${save_name}
```
For example:
```bash
bash inference.sh 1 inference "A lion is attacking." "data/vdm/examples/images" "051a2a7ea842426f825e128fef3bf92b" "vdm/inference"
```
You can find the outputs in `outputs/vdm/inference` folder. 