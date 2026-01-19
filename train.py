from PIL import Image
import os
import random
from model import Model
import sys
import math
import time

def test_photo_unique(all, test):
    for epi_class in all:
        if test in epi_class:
            return False
    return True

def make2DPhoto(seqImData):
    toCopy = []
    lastRow = []
    for j,item in enumerate(seqImData):
        lastRow.append(item)
        if len(lastRow) == 28:
            toCopy.append(lastRow)
            lastRow = []
    return toCopy

def add_component_wise(vec1, vec2):
    if len(vec2) == 0: return vec1
    new = []
    for i in range(len(vec1)):
        new.append(vec1[i] + vec2[i])
    return new

def scalar_to_vec(scalar, vec):
    for i in range(len(vec)):
        vec[i]*=scalar
    return vec


if __name__ == "__main__":
    classes = os.listdir("omniglot/processed/train_processed")
    ITERATIONS = 1000
    model = Model()

    for it in range(ITERATIONS):
        episode_classes = random.sample(classes, 15)

        all_photos = []
        #These are the "train"
        for epi_class in episode_classes:
            chars = os.listdir(f"omniglot/processed/train_processed/{epi_class}")
            selected = random.sample(chars, 20)
            all_photos.append(selected)

        #test for training without replacement
        total_photos = len(episode_classes) * 80
        test = []
        for i in range(5):
            while True:
                idx = math.floor(total_photos * random.random())
                which_class = idx//80
                image_num = idx - which_class*80

                chars = os.listdir(f"omniglot/processed/train_processed/{episode_classes[which_class]}")
                if test_photo_unique(all_photos, chars[image_num]):
                    test.append(chars[image_num])
                    break

        #call eval for all train
        for i, epi_class in enumerate(all_photos):
            class_mean = []
            #evaluate the image and calcualte the euclidean distance mean
            sum = 0
            for img in epi_class:
                originalTime = time.time_ns()
                imData = Image.open(f"omniglot/processed/train_processed/{episode_classes[i]}/{img}").resize((28,28)).convert("1").getdata()
                seqImData = list(imData)
                orderedImData = make2DPhoto(seqImData)
                output_embed = model.evaluate(orderedImData)
                class_mean = add_component_wise(output_embed, class_mean)

                #other logging metrics
                sum += time.time_ns() - originalTime
            print(f"Average seconds per evaluation: {sum/len(epi_class)/pow(10, 9)}")

            #divide by total imgs
            class_mean = scalar_to_vec(1.0/len(epi_class), class_mean)
            print(class_mean)
            sys.exit()

