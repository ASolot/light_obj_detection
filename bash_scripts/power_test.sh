
cd $HOME/MThesis/repos/mine/light_obj_detection/SolotNet

PYTHONPATH="" 

echo "Start DLA_C_1024\n"
echo "Infer: 1024\n"
python3.5 evaluate.py ctdet --exp_id visdrone_dla_c_1024 --dataset visdrone --input_res 1024 --resume
sleep 10

echo "Infer: 512\n"
python3.5 evaluate.py ctdet --exp_id visdrone_dla_c_1024 --dataset visdrone --input_res 512 --resume
sleep 10

echo "Infer: orig\n"
python3.5 evaluate.py ctdet --exp_id visdrone_dla_c_1024 --dataset visdrone --keep_res--resume
sleep 10



sleep 10
echo "Start DLA_C_512\n"
echo "Infer: 1024\n"
python3.5 evaluate.py ctdet --exp_id visdrone_dla_c_512 --dataset visdrone --input_res 1024 --resume
sleep 10

echo "Infer: 512\n"
python3.5 evaluate.py ctdet --exp_id visdrone_dla_c_512 --dataset visdrone --input_res 512 --resume
sleep 10

echo "Infer: orig\n"
python3.5 evaluate.py ctdet --exp_id visdrone_dla_c_512 --dataset visdrone --keep_res --resume
sleep 10



sleep 10
echo "START ESPNET_C_1024\n"
echo "Infer: 1024\n"
python3.5 evaluate.py ctdet --exp_id visdrone_esp_c_1024 --arch espnetv2 --dataset visdrone --input_res 1024 --resume
sleep 10

echo "Infer: 512\n"
python3.5 evaluate.py ctdet --exp_id visdrone_esp_c_1024 --arch espnetv2 --dataset visdrone --input_res 512 --resume
sleep 10

echo "Infer: orig\n"
python3.5 evaluate.py ctdet --exp_id visdrone_esp_c_1024 --arch espnetv2 --dataset visdrone --keep_res --resume
sleep 10




sleep 10
echo "START ESPNET_C_512\n"
echo "Infer: 1024\n"
python3.5 evaluate.py ctdet --exp_id visdrone_esp_c_512 --arch espnetv2 --dataset visdrone --input_res 1204 --resume
sleep 10

echo "Infer: 512\n"
python3.5 evaluate.py ctdet --exp_id visdrone_esp_c_512 --arch espnetv2 --dataset visdrone --input_res 512 --resume
sleep 10

echo "Infer: orig\n"
python3.5 evaluate.py ctdet --exp_id visdrone_esp_c_512 --arch espnetv2 --dataset visdrone --keep_res --resume
sleep 10





sleep 10
echo "START_RESNET_1024\n"
echo "Infer: 1024\n"
python3.5 evaluate.py ctdet --exp_id visdrone_res_1024 --arch resdcn_18 --dataset visdrone --input_res 1024 --resume
sleep 10

echo "Infer: 512\n"
python3.5 evaluate.py ctdet --exp_id visdrone_res_1024 --arch resdcn_18 --dataset visdrone --input_res 512 --resume
sleep 10

echo "Infer: orig\n"
python3.5 evaluate.py ctdet --exp_id visdrone_res_1024 --arch resdcn_18 --dataset visdrone --keep_res --resume
sleep 10



