# Track your pullups sessions

Just let the program run while doing your workout, it will detect your pullup sets and track your movement.<br />
It may break if multiple many people are detected in the same time (still, can be fixed).<br />
Pull up detection was trained on a resnet pretrained model finetuned on a custom dataset of personnal pullups videos and google-image's result typing "sport pull ups".
Originaly it was supposed to also perform keypoint detection, but not enought data has been labeled.<br />
Keypoints detection is performed using mediapipe library.<br />

<p align="center">
  <img src="ReadmePic.png" width="100%" title="">
</p>
