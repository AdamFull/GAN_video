import os, cv2, re
from util import Visualization

vis = Visualization()

def load_images(path):
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    images = []
    for i in range(len(files)):
        vis.print_progress_bar(i+1, len(files), label="PROCESSING frame%d" % i, points=70)
        image = cv2.imread(files[i])
        images.append(image)
    size = image.shape[:2]
    print("")
    return images, size


def images_to_video(path, frame_rate=30):
    print("Making video...")
    final_path = os.path.join(os.path.dirname(path), "final")
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    frames_array, size = load_images(path)

    out = cv2.VideoWriter(os.path.join(final_path, "final.avi"),cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, size)
    font = cv2.QT_FONT_BLACK
    bottomLeftCornerOfText = (0,50)
    fontScale = 2
    fontColor = (0,0,0)
    lineType = 2

    for i in range(len(frames_array)):
        image = frames_array[i]
        cv2.putText(image,'Epoch %s' % str(i+1), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
        out.write(image)
        vis.print_progress_bar(i+1, len(frames_array), label="PROCESSING frame%d" % i, points=70)
    out.release()
    del(frames_array)
    print("")

if __name__ == "__main__":
    path = input("Folder path: ")
    images_to_video(path, 30)