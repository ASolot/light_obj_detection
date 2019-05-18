from PIL import Image
import os

from tqdm import tqdm

DATASET_ROOT_PATH = "/home/alex/MThesis/datasets/VISDRONE"

SOURCE = "/Images"

DESTINATION_WIDTH  = 512
DESTINATION_HEIGHT = 512

DESTINATION = "/Images-" + str(DESTINATION_WIDTH) + 'x' + str(DESTINATION_HEIGHT)

TRAIN_DATA = "/VisDrone2018-DET-train"
TEST_DATA  = "/VisDrone2018-DET-val" 

ANNOTATIONS = "/annotations"
IMAGES = "/images"

SOURCE_TEST_DIRECTORY       = DATASET_ROOT_PATH + SOURCE + TEST_DATA
SOURCE_TRAIN_DIRECTORY      = DATASET_ROOT_PATH + SOURCE + TRAIN_DATA

DESTINATION_TEST_DIRECTORY  = DATASET_ROOT_PATH + DESTINATION + TEST_DATA
DESTINATION_TRAIN_DIRECTORY = DATASET_ROOT_PATH + DESTINATION + TRAIN_DATA

def annotation_resize(source_file, destination_file, width_shrink, height_shrink):
    
    f = open(source_file)
    g = open(destination_file, 'w')
        
    lines = f.readlines()
            
    for line in lines:
        data = line.split(',')
        processed = str(round(int(data[0])*width_shrink))
        processed += ','
        processed += str(round(int(data[1])*height_shrink))
        processed += ','
        processed += str(round(int(data[2])*width_shrink))
        processed += ','
        processed += str(round(int(data[3])*height_shrink))
        processed += ','
        processed += str(int(data[4]))
        processed += ','
        processed += str(int(data[5]))
        processed += '\n'
        g.write(processed)

def images_pad_resize(source_path, destination_path): 

    images_source_path              = source_path + IMAGES
    images_destination_path         = destination_path + IMAGES
    annotations_source_path         = source_path + ANNOTATIONS
    annotations_destination_path    = destination_path + ANNOTATIONS

    # Create the dirs 
    if not os.path.exists(images_destination_path):
        os.makedirs(images_destination_path)
        print("Directory " , images_destination_path ,  " Created ")
    else:    
        print("Directory " , images_destination_path ,  " already exists")

    if not os.path.exists(annotations_destination_path):
        os.makedirs(annotations_destination_path)
        print("Directory " , annotations_destination_path ,  " Created ")
    else:    
        print("Directory " , annotations_destination_path ,  " already exists")

    filenames = [ x for x in os.listdir(images_source_path) if x.endswith('.jpg')]
    x = 0
    for filename in tqdm(filenames, desc="Reading files"):
        # if x > 2: 
        #     continue
        x = x+1
        im_path         = images_source_path + '/' + filename
        im_dest_file    = images_destination_path + '/' + filename
        im              = Image.open(im_path)

        old_size = im.size

        # the outputs are square
        ratio = float(DESTINATION_WIDTH)/max(old_size)

        new_size = tuple([int(x*ratio) for x in old_size])
        im = im.resize(new_size, Image.ANTIALIAS)

        # create a new image and paste the resized on it
        new_im = Image.new("RGB", (DESTINATION_WIDTH, DESTINATION_WIDTH))
        new_im.paste(im)

        # write image to new destination
        new_im.save(im_dest_file)

        # process annotations
        an_filename = filename.split('.')[0]

        an_source_file = annotations_source_path + '/' + an_filename + '.txt'
        an_dest_file   = annotations_destination_path + '/' + an_filename + '.txt'
        
        annotation_resize(an_source_file, an_dest_file, ratio, ratio)

def image_crop_and_resize(source_path, destination_path):
    
    images_source_path              = source_path + IMAGES
    images_destination_path         = destination_path + IMAGES
    annotations_source_path         = source_path + ANNOTATIONS
    annotations_destination_path    = destination_path + ANNOTATIONS

    # Create the dirs 
    if not os.path.exists(images_destination_path):
        os.makedirs(images_destination_path)
        print("Directory " , images_destination_path ,  " Created ")
    else:    
        print("Directory " , images_destination_path ,  " already exists")

    if not os.path.exists(annotations_destination_path):
        os.makedirs(annotations_destination_path)
        print("Directory " , annotations_destination_path ,  " Created ")
    else:    
        print("Directory " , annotations_destination_path ,  " already exists")

    filenames = [ x for x in os.listdir(images_source_path) if x.endswith('.jpg')]

    for filename in tqdm(filenames, desc="Reading files"):
        # if x > 2: 
        #     continue
        x = x+1
        im_path         = images_source_path + '/' + filename
        im_dest_file    = images_destination_path + '/' + filename
        im              = Image.open(im_path)

        old_size = im.size

        # the outputs are square
        ratio = float(DESTINATION_WIDTH)/max(old_size)

        new_size = tuple([int(x*ratio) for x in old_size])
        im = im.resize(new_size, Image.ANTIALIAS)

        # create a new image and paste the resized on it
        new_im = Image.new("RGB", (DESTINATION_WIDTH, DESTINATION_WIDTH))
        new_im.paste(im)

        # write image to new destination
        new_im.save(im_dest_file)

        # process annotations
        root_filename = filename.split('.')[0]
        
        an_filename = root_filename + '.txt'
        im_filename = root_filename + '.jpg' 

        an_source_file = annotations_source_path + '/' + an_filename 
        an_dest_file   = annotations_destination_path + '/' + an_filename
        
        annotation_resize(an_source_file, an_dest_file, ratio, ratio)

def main():

    print ("Resizing to: ", DESTINATION_WIDTH, 'x', DESTINATION_HEIGHT)

    print ("Resizing Test dataset\n\n")
    images_pad_resize(SOURCE_TEST_DIRECTORY, DESTINATION_TEST_DIRECTORY)

    print ("Resizing Valid dataset\n\n")
    images_pad_resize(SOURCE_TRAIN_DIRECTORY, DESTINATION_TRAIN_DIRECTORY)




if __name__ == '__main__':
    main()