import torch
import torchvision
import PullupModel as Pullup

import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

MODE = "webcam"
NMAX_LOGITS = 6
TRUE_COLOR = mp_drawing.DrawingSpec(color=mp_drawing.GREEN_COLOR)
FALSE_COLOR = mp_drawing.DrawingSpec(color=mp_drawing.RED_COLOR)
last_logits = []


def init_model(path):
    model = Pullup.create_model()
    model = Pullup.load_weights(model, path)
    model.eval()
    return model


def inference(image):
    image = vanilla_transform(image).to(device).unsqueeze(0)
    with torch.no_grad():
        pred = model(image)
    res = float(torch.sigmoid(pred[1].to("cpu")).numpy()[0])
    return res


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
vanilla_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(norm_mean, norm_std)])
model = init_model("output/model/model-score0.96-f1_10.9-f1_20.99.pt")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model = model.to(device)


def process_frame(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(image)
    classif_logit = inference(image)

    global last_logits
    last_logits.append(classif_logit)
    if len(last_logits) > NMAX_LOGITS:
        last_logits = last_logits[1:]
    if np.mean(last_logits) >= 0.5:
        drawing_spec = TRUE_COLOR
        bar_color = (0, 180, 0)
    else:
        drawing_spec = FALSE_COLOR
        bar_color = (0, 0, 180)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        pose_result.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=drawing_spec)
    image = cv2.rectangle(image, (0, 0), (50, 200), (120, 120, 120), -1)
    image = cv2.rectangle(image, (0, 200 - int(np.mean(last_logits) * 200)), (50, 200), bar_color, -1)

    return image, pose_result, classif_logit


def add_result(res_list, time, pose_result, classif_logit):
    added_data = False
    if hasattr(pose_result.pose_landmarks, 'landmark'):
        right_shoulder = pose_result.pose_landmarks.landmark[11]
        left_shoulder = pose_result.pose_landmarks.landmark[12]
        body_x_landmark = (right_shoulder.x + left_shoulder.x) / 2
        body_y_landmark = (right_shoulder.y + left_shoulder.y) / 2

        right_hand = pose_result.pose_landmarks.landmark[15]
        left_hand = pose_result.pose_landmarks.landmark[16]
        between_hands_x_landmark = (right_hand.x + left_hand.x) / 2
        between_hands_y_landmark = (right_hand.y + left_hand.y) / 2

        res_list.append({'elapsed_time': time,
                         'body_x_landmark': body_x_landmark,
                         'body_y_landmark': body_y_landmark,
                         'between_hands_x_landmark': between_hands_x_landmark,
                         'between_hands_y_landmark': between_hands_y_landmark,
                         'classif_logit': classif_logit})
        added_data = True
    return res_list, added_data


def save(results_list):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M")
    print("Session end at", dt_string)
    print("Saving ...")
    df = pd.DataFrame(res_list)
    df["date_time"] = now.strftime(dt_string)
    df.to_csv(f"output/sessions_data/session_{dt_string}.csv", index=False)
    print("Saved")


if __name__ == "__main__":
    res_list = []

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        if MODE == 'webcam':
            # For webcam input:
            cap = cv2.VideoCapture(0)
            previous_time = time.time()
            while cap.isOpened():
                success, image = cap.read()
                image, pose_result, classif_logit = process_frame(image)

                elapsed_time = time.time() - previous_time
                fps = str(int(1 / elapsed_time))
                res_list, added_data = add_result(res_list, elapsed_time, pose_result, classif_logit)
                if added_data:
                    previous_time = time.time()

                image = cv2.flip(image, 1)
                cv2.putText(image, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('MediaPipe x PullUpClassif Pose', image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

        elif MODE == "video":
            vidcap = cv2.VideoCapture("video_test.mp4")
            success, image = vidcap.read()
            count = 0
            previous_time = 0
            while success:
                image, pose_result, classif_logit = process_frame(image)

                elapsed_time = time.time() - previous_time
                fps = str(int(1 / elapsed_time))
                res_list, added_data = add_result(res_list, elapsed_time, pose_result, classif_logit)
                if added_data:
                    previous_time = time.time()

                cv2.putText(image, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('MediaPipe x PullUpClassif Pose', image)

                success, image = vidcap.read()
                count += 1
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    cap.release()
    save(res_list)
