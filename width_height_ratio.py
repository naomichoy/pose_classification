import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time, sys
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import argparse
import os.path

# CSI camera setup
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


'''
hnum: 0 based human index
kpoint : keypoints (float type range : 0.0 ~ 1.0 ==> later multiply by image width, height
'''
def get_keypoint(humans, hnum, peaks):
    #check invalid human index
    kpoint = []
    human = humans[0][hnum]
    # print(human)
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
            # print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
        else:    
            peak = (j, None, None)
            kpoint.append(peak)
            # print('index:%d : None %d'%(j, k) )
    return kpoint


def fall_body_ratio(keypoints):
    keypoints_y = []
    keypoints_x = []
    for i in range(len(keypoints)):
        keypoints_y.append(keypoints[i][1])
        keypoints_x.append(keypoints[i][0])
    y_max = max(keypoints_y)
    y_min = min(keypoints_y)
    x_max = max(keypoints_x)
    x_min = min(keypoints_x)
    body_ratio = (y_max-y_min)/(x_max-x_min)
    if body_ratio < 1:
        return True
    else:
        return False


parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
parser.add_argument('--model', type=str, default='resnet', help = 'resnet or densenet' )
args = parser.parse_args()

# load body part specification
with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])


# choose model
if 'resnet' in args.model:
    print('------ model = resnet--------')
    MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
    OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 224
    HEIGHT = 224

else:    
    print('------ model = densenet--------')
    MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'
    OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
    model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 256
    HEIGHT = 256

# load trt optimised model, if exist not create one
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
if os.path.exists(OPTIMIZED_MODEL) == False:
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def execute(img, t):
    global isFallRatio
    color = (0, 255, 0) # <-- for drawing circle points
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    fps = 1.0 / (time.time() - t)

    keypoints_y = []
    keypoints_x = []

    # print(counts[0])
    if counts[0] == 1: # only extract the key points if exactly one person is detected

        keypoints = get_keypoint(objects, 0, peaks)
        # isFallRatio = fall_body_ratio(keypoints)

        if keypoints[0][1] or keypoints[17][1]:
            head = keypoints[0]
            neck = keypoints[17]
            Lhip = keypoints[11]
            Rhip = keypoints[12]
            Lshoulder = keypoints[5]
            Rshoulder = keypoints[6]

            if head[1]: 
                head_y = head[1] * HEIGHT # index 1 = height
                head_y_list.append(head_y)


    # reference code
    # for i in range(counts[0]):
    #     print("Human index:%d "%( i ))
    #     keypoints = get_keypoint(objects, i, peaks)
    #     print(keypoints[0])
        for j in range(len(keypoints)):
            # print(keypoints[j])
            if keypoints[j][1]:
                x = round(keypoints[j][2] * WIDTH )
                y = round(keypoints[j][1] * HEIGHT)
        #         x = round(keypoints[j][2] * WIDTH * X_compress) # incorrect adjustment, compress factor not needed
        #         y = round(keypoints[j][1] * HEIGHT * Y_compress)
                cv2.circle(img, (x, y), 3, color, 1)
                cv2.putText(img , "%d" % int(keypoints[j][0]), (x + 5, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

                keypoints_y.append(keypoints[j][1])
                keypoints_x.append(keypoints[j][2])

        if len(keypoints_y) > 0:
            y_max = int(max(keypoints_y) * HEIGHT)
            y_min = int(min(keypoints_y) * HEIGHT)
            x_max = int(max(keypoints_x) * WIDTH)
            x_min = int(min(keypoints_x) * WIDTH)
            # print("max and min:", y_max, y_min, x_max, x_min)
            cv2.line(img, (x_min, y_max), (x_min,y_min), (255, 0, 0), 1) # (start x,y) , (end x,y)
            cv2.line(img, (x_min, y_min), (x_max,y_min), (255, 0, 0), 1)
            cv2.line(img, (x_max, y_max), (x_max,y_min), (255, 0, 0), 1)
            cv2.line(img, (x_min, y_max), (x_max,y_max), (255, 0, 0), 1)

            if (x_max-x_min) != 0:
                body_ratio = (y_max-y_min)/(x_max-x_min)
                print("ratio:", body_ratio)
                if body_ratio < 1:
                    isFallRatio = True
                else:
                    isFallRatio = False
    
    # draw_objects(img, counts, objects, peaks)

    # cv2.putText(img , "FPS: %f" % (fps), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    # cv2.imshow('frame', src)
    print("FPS:%f "%(fps))
    return img
    # out_video.write(src)


# initialise camera stream
cap = cv2.VideoCapture(0) # usb camera
# cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0),cv2.CAP_GSTREAMER) # CSI camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret_val, img = cap.read()
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out_video = cv2.VideoWriter('/tmp/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (640, 480))
count = 0

X_compress = 640.0 / WIDTH * 1.0
Y_compress = 480.0 / HEIGHT * 1.0

if cap is None:
    print("Camera Open Error")
    sys.exit(0)

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

# main function

head_y_list = []
# coordinates = []

FALL_THRESHOLD = 60
ALERT_THRESHOLD = 5
STAND_THRESHOLD = 25

fallFlag = 0    # 0 no fall, 1 warning, 2 alarm
fall_start = 0
fall_time = 0

isFallRatio = False

while (True):  #cap.isOpened() and count < 500:
    # read camera frame
    t = time.time()
    ret, frame = cap.read()
    if ret == False:
        print("Camera read Error")
        break
    imgg = cv2.resize(frame, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    # feed frame into OpenPose
    output = execute(imgg, t)

    print("ratio:", isFallRatio)
    if len(head_y_list) > 5: 
        hDiff = head_y_list[4]-head_y_list[0]
        # hDiff = coordinates[4][0]-coordinates[0][0]
        # nDiff = coordinates[4][1]-coordinates[0][1]
        # if hDiff > FALL_THRESHOLD or nDiff > FALL_THRESHOLD + 20: # neck fall triggered sitting
        if hDiff > FALL_THRESHOLD and isFallRatio:
            fallFlag = 1
            fall_start = time.time()
            # print("flag")
        if fallFlag == 1:
            # if head_y_list[4] < HEIGHT / 2 and hDiff > STAND_THRESHOLD:
            # if coordinates[4][0] < HEIGHT / 2 and hDiff > STAND_THRESHOLD:
            # if not isFallRatio:
            if not isFallRatio and hDiff > STAND_THRESHOLD:
                print("stood up")
                fallFlag = 0
            else:
                fall_time = time.time()

        if fall_time - fall_start > ALERT_THRESHOLD:
            fallFlag = 2
        
        head_y_list.pop(0)
        # coordinates.pop(0)
    
    print("last frames", head_y_list)
    # print(coordinates)
    print("fallFlag:", fallFlag)
   
    count += 1
    print(count)
    # for demo purpose
    if count > 500:
        fallFlag = 0
        count = 0
        fall_start = 0
        fall_time = 0

    cv2.imshow('frame',output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break


cv2.destroyAllWindows()
# out_video.release()
cap.release()
