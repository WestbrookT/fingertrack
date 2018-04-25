
import ibench, random
from tqdm import tqdm

datasets = [
    'I_Chinesebook',
    'I_ClassRoom',
    'I_BasketBallField',
    'I_Avenue',
    'I_ComputerScreen',
    'I_EastCanteen'
]

path = './images/{}/{}'



def parse_datasets(keep_prob=.2, max_amount=1000):

    images = []
    points = []
    for dataset in datasets:

        f = open('./txt/{}_label.txt'.format(dataset))
        

        for line in tqdm(f):
            if random.random() < keep_prob:
                continue
            line = line.split()
            if len(line) > 0:

                img_path = path.format(dataset, line[0])

                img = ibench.to_PIL(img_path)

                images.append(img_path)
            
                ftx = int(img.width * float(line[5]))
                fty = int(img.height * float(line[6]))

                
                jntx = int(img.width * float(line[7]))
                jnty = int(img.height * float(line[8]))

                
                tjntx = int(img.width * float(line[9]))
                tjnty = int(img.height * float(line[10]))



                points.append([(ftx, fty), (jntx, jnty), (tjntx, tjnty)])
            if len(images) > max_amount:
                break
            
    print('Paths found:', len(images))
    return images, points


