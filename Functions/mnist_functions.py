from pickle import load
import numpy as np
import cv2
import math
from torchvision import transforms
import PIL
from PIL import Image
import imutils
from io import BytesIO



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
model = load(open('Models/mnist_model.pkl', 'rb'))

def returnPaddedImage(dim, image):
  pixelsToAdd = 30
  [x,y,w,h] = dim
  rawImage = image [y : y + h, x : x + w ]

  newPaddedImage = []

  if h > w :
    newPaddedImage = [ [[0, 0, 0]]*(h + 2*pixelsToAdd) ]*pixelsToAdd
    leftPixelsAddCount =  ((h + 2*pixelsToAdd) - w) // 2
    rightPixelsAddCount = (h + 2*pixelsToAdd) - w - leftPixelsAddCount
    for row in rawImage:
      part1 = [[0, 0, 0]]*leftPixelsAddCount
      part2 = [[0, 0, 0]]*rightPixelsAddCount
      part1.extend(row)
      part1.extend(part2)
      newPaddedImage.append(part1)
      # print(len(part1))
    newPaddedImage.extend( [ [[0, 0, 0]]*(h + 2*pixelsToAdd) ]*pixelsToAdd )

  else:
    print("YAYAYAYAYAAYAY")
    newPaddedImage = [ [[0, 0, 0]]*(w + 2*pixelsToAdd) ]* ((len(rawImage[0]) + 2*pixelsToAdd - h) // 2)
    leftPixelsAddCount =  pixelsToAdd
    rightPixelsAddCount = pixelsToAdd
    for row in rawImage:
      part1 = [[0, 0, 0]]*leftPixelsAddCount
      part2 = [[0, 0, 0]]*rightPixelsAddCount
      part1.extend(row)
      part1.extend(part2)
      newPaddedImage.append(part1)
    newPaddedImage.extend( [ [[0, 0, 0]]*(w + 2*pixelsToAdd) ] * math.ceil((len(rawImage[0]) + 2*pixelsToAdd - h) / 2) )

  return np.asarray(newPaddedImage, dtype=np.float32)

def predictSomething(img):
    # with torch.no_grad():
    logps = model(img)
    # print(model)

    # ps = torch.exp(logps)
    # probab = list(ps.numpy()[0])
    # convert to exp without using pytorch
    # probab = np.exp(logps.numpy()[0])
    probab = logps.detach().exp().numpy()
    print(probab)
    i = 0
    for p in probab[0]:
        if p > 0.5:
            return i
        i = i + 1
    # print("Predicted Digit =", probab.index(max(probab)))
    # # view_classify(img.view(1, 28, 28), ps)
    # return probab.index(max(probab))


def handleMnistPrediction(file):
    file_obj = BytesIO(file)
    pil_obj = Image.open(file_obj)
    image = cv2.cvtColor(np.array(pil_obj), cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    i = 0
    arrForSort = []
    for c in cnts:
        [x, y, w, h] = cv2.boundingRect(c)
        arrForSort.append([x, y, w, h])
        i = i + 1
    # arrForSort.sort(key=lambda x: x[1])
    arrForSort.sort(key=lambda x: x[0])
    # print(arrForSort)

    ans = ""

    for xyz in arrForSort:
        [x, y, w, h] = xyz
        # scaledImage = (cv2.resize(image [y : y + h, x : x + w ], (28, 28), interpolation=cv2.INTER_AREA  ) )
        scaledImage = (cv2.resize(returnPaddedImage(xyz, image), (28, 28), interpolation=cv2.INTER_AREA))
        pilImage = PIL.Image.fromarray(cv2.cvtColor(scaledImage, cv2.COLOR_BGR2GRAY))
        ptImage = transform(pilImage).unsqueeze(0).view(1, 784)
        ans += str(predictSomething(ptImage))

    # ans = predictSomething(transform(pil_obj).unsqueeze(0).view(1, 784))

    # print(image)
    # print("Hi")
    return ans
