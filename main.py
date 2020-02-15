import os

from dataset_creator import dataset
from gan import GAN

width, heigth = 80, 80

self_path = os.path.dirname(os.path.abspath(__file__))
temp_path = os.path.join(self_path, "temp")



    # print("Video saved to %s" % final_path)

if __name__ == "__main__":
    #file_path = input("write path to video:")
    dataset_data = dataset("file_path", temp_path, width, heigth)
    data = dataset_data.prep_imgs(os.path.join(self_path, "datasets/for_tests"))
    generative_net = GAN(buff_size=33136, batch_size=8, epochs=1000, imgs_size=(width, heigth))
    generative_net.train(data)
    #dataset_data.prepare()
    ##shutil.rmtree(temp_path)
