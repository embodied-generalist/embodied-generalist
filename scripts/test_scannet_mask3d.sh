python launch.py --name leo_tuning \
                 --qos lv0b \
                 --mem_per_gpu 100 \
                 --time 48 \
                 --config configs/default.yaml \
                 --port 2110 \
                 --gpu_per_node 4 \
                 --num_nodes 1 \
                 --partition HGX \
                 mode=eval \
                 task=test_scannet \
                 note=test_scannet_mask3d \
                 data.scan2cap.pc_type=pred \
                 data.scanqa.pc_type=pred \
                 data.sqa3d.pc_type=pred \
                 pretrained_ckpt_path={TBD} \
