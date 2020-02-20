import os

from dataset_creator import dataset
from gan import GAN

width, heigth = 128, 128

self_path = os.path.dirname(os.path.abspath(__file__))
temp_path = os.path.join(self_path, "temp")



    # print("Video saved to %s" % final_path)

def main():
    comand = input("Select what you want( train, start_bot, clean): ")
    if comand == "train":
        datasets = os.path.join(self_path, "datasets")
        dirs = [dI for dI in os.listdir(datasets) if os.path.isdir(os.path.join(datasets,dI))]
        comand = input("Select dataset(%s):" % ','.join(dirs))
        dataset_data = dataset("file_path", temp_path, width, heigth)
        data = dataset_data.prep_imgs(os.path.join(self_path, "datasets/%s" % comand))
        #generative_net = GAN(buff_size=, batch_size=4, epochs=5000, imgs_size=(width, heigth))
        #generative_net.train(data)
        
        

if __name__ == "__main__":
    #file_path = input("write path to video:")
    dataset_data = dataset("file_path", temp_path, width, heigth)
    data = dataset_data.prep_imgs(os.path.join(self_path, "datasets/big_anime"))
    generative_net = GAN(buff_size=10035, batch_size=16, epochs=5000, imgs_size=(width, heigth))
    generative_net.train(data)
