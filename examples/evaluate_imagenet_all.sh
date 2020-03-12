#!/bin/sh

CUDA_VISIBLE_DEVICES=0 screen -dLm bash -c "echo 0; \
python evaluate_imagenet.py --model_name=ResNet50 --eval_image_size=224 --normalize=1; \
python evaluate_imagenet.py --model_name=ResNet101 --eval_image_size=224 --normalize=1; \
python evaluate_imagenet.py --model_name=ResNet152 --eval_image_size=224 --normalize=1; \
python evaluate_imagenet.py --model_name=ResNeXt101c64 --eval_image_size=224 --normalize=3; \
python evaluate_imagenet.py --model_name=Inception3 --eval_image_size=299 --normalize=2; \
python evaluate_imagenet.py --model_name=MobileNet100 --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=MobileNet35v2 --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=MobileNet50v2 --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=MobileNet140v2 --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=MobileNet75v3large --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=MobileNet75v3small --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=EfficientNetB0 --eval_image_size=224 --normalize=3; \
python evaluate_imagenet.py --model_name=EfficientNetB1 --eval_image_size=240 --normalize=3; \
python evaluate_imagenet.py --model_name=EfficientNetB5 --eval_image_size=456 --normalize=3 --batch_size=100; \
python evaluate_imagenet.py --model_name=DenseNet121 --eval_image_size=224 --normalize=3; \
python evaluate_imagenet.py --model_name=VGG19 --eval_image_size=224 --normalize=1; \
python evaluate_imagenet.py --model_name=NASNetAlarge --eval_image_size=331 --normalize=2 --batch_size=100"

CUDA_VISIBLE_DEVICES=1 screen -dLm bash -c "echo 1; \
python evaluate_imagenet.py --model_name=ResNet50v2 --eval_image_size=299 --normalize=2; \
python evaluate_imagenet.py --model_name=ResNet101v2 --eval_image_size=299 --normalize=2; \
python evaluate_imagenet.py --model_name=ResNet152v2 --eval_image_size=299 --normalize=2; \
python evaluate_imagenet.py --model_name=Inception4 --eval_image_size=299 --normalize=2; \
python evaluate_imagenet.py --model_name=InceptionResNet2 --eval_image_size=299 --normalize=2; \
python evaluate_imagenet.py --model_name=MobileNet75 --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=MobileNet75v2 --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=MobileNet100v2 --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=MobileNet130v2 --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=MobileNet100v3large --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=MobileNet100v3small --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=EfficientNetB2 --eval_image_size=260 --normalize=3; \
python evaluate_imagenet.py --model_name=EfficientNetB3 --eval_image_size=300 --normalize=3; \
python evaluate_imagenet.py --model_name=EfficientNetB6 --eval_image_size=528 --normalize=3 --batch_size=50; \
python evaluate_imagenet.py --model_name=DenseNet169 --eval_image_size=224 --normalize=3; \
python evaluate_imagenet.py --model_name=VGG16 --eval_image_size=224 --normalize=1; \
python evaluate_imagenet.py --model_name=PNASNetlarge --eval_image_size=331 --normalize=2 --batch_size=100"

CUDA_VISIBLE_DEVICES=2 screen -dLm bash -c "echo 2; \
python evaluate_imagenet.py --model_name=ResNet200v2 --eval_image_size=224 --normalize=3; \
python evaluate_imagenet.py --model_name=ResNeXt50 --eval_image_size=224 --normalize=3; \
python evaluate_imagenet.py --model_name=ResNeXt101 --eval_image_size=224 --normalize=3; \
python evaluate_imagenet.py --model_name=WideResNet50 --eval_image_size=224 --normalize=5; \
python evaluate_imagenet.py --model_name=Inception1 --eval_image_size=224 --normalize=4; \
python evaluate_imagenet.py --model_name=Inception2 --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=MobileNet25 --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=MobileNet50 --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=MobileNet100v3largemini --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=MobileNet100v3smallmini --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=EfficientNetB4 --eval_image_size=380 --normalize=3 --batch_size=100; \
python evaluate_imagenet.py --model_name=EfficientNetB7 --eval_image_size=600 --normalize=3 --batch_size=50; \
python evaluate_imagenet.py --model_name=DenseNet201 --eval_image_size=224 --normalize=3; \
python evaluate_imagenet.py --model_name=NASNetAmobile --eval_image_size=224 --normalize=2; \
python evaluate_imagenet.py --model_name=SqueezeNet --eval_image_size=224 --normalize=4"

cat screenlog.0 | grep "^|" | sort | uniq
