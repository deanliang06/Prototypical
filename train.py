from PIL import Image
import os
import random
from model import Model
import sys
import math

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

def euclidean_distance(vec1, vec2):
    sum_of_squares = 0
    for i in range(len(vec1)):
        sum_of_squares+=(vec1[i]-vec2[i])**2
    return sum_of_squares**(1/2)

def make_photo_2D(link):
    imData = Image.open(link).resize((28,28)).convert("1").getdata()
    seqImData = list(imData)
    orderedImData = make2DPhoto(seqImData)
    return orderedImData


if __name__ == "__main__":
    classes = os.listdir("omniglot/processed/train_processed")
    LEARNING_RATE = 0.005
    ITERATIONS = 1000
    model = Model(LEARNING_RATE)

    for it in range(ITERATIONS):
        episode_classes = random.sample(classes, 15)

        all_photos = []
        #These are the "train"
        for epi_class in episode_classes:
            chars = os.listdir(f"omniglot/processed/train_processed/{epi_class}")
            selected = random.sample(chars, 20)
            selected_address = []
            for one in selected:
                selected_address.append(f"omniglot/processed/train_processed/{epi_class}/{one}")
            all_photos.append({"imgs": selected_address, "class": epi_class})

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
                    test.append({"img": f"omniglot/processed/train_processed/{episode_classes[which_class]}/{chars[image_num]}", "class": episode_classes[which_class]})
                    break

        #call eval for all train
        means = {}
        for i, epi_class_object in enumerate(all_photos):
            the_class = epi_class_object["class"]
            epi_class = epi_class_object["imgs"]
            class_mean = []
            #evaluate the image and calcualte the euclidean distance mean
            sum = 0
            for img in epi_class:
                orderedImData = make_photo_2D(img)
                output_embed = model.evaluate(orderedImData)
                class_mean = add_component_wise(output_embed, class_mean)

            #divide by total imgs
            class_mean = scalar_to_vec(1.0/len(epi_class), class_mean)
            means[the_class] = class_mean

        loss = 0
        for img_object in test:
            img = img_object["img"]
            target = means[img_object["class"]]
            orderedImData = make_photo_2D(img)
            prediction = model.evaluate(orderedImData)

            #now compute loss and gradients
            #loss

            loss+=model.compute_loss(prediction, target, means)
            #now calculate gradients
            model.apply_gradients(means, prediction, img_object["class"])

        print(f"Iteration #{it + 1} Average Loss: {loss}")
            

