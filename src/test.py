import torch
import clip
import PIL
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import rospy
from pygame import mixer

device = "cuda" if torch.cuda.is_available() else "cpu" 
model, preprocess = clip.load("ViT-B/32", device=device)
bridge = CvBridge()

def beepsound():
    freq = 2000    # range : 37 ~ 32767
    dur = 1000     # ms
    ws.Beep(freq, dur) # winsound.Beep(frequency, duration)

def perception(data):
    print("hell")
    image = preprocess(PIL.Image.fromarray(bridge.imgmsg_to_cv2(data))).unsqueeze(0).to(device)
    text = clip.tokenize(["knife","safe"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
    if(probs[0,0]>0.8):
        mixer.init() 
        sound=mixer.Sound("/home/jiwoo/Documents/catkin_ws/src/clip_/src/bell.wav")
        sound.play()

def perception_node():
    rospy.init_node('perception', anonymous=True)
    rospy.Subscriber("/camera",Image,perception)
    rospy.spin()

if __name__ == '__main__':
    try: perception_node();
        
    except rospy.ROSInterruptException:
        pass