import pygame, ibench, numpy as np
import time, pygame.camera, data
from model import model_v1, model_v2

pygame.init()

size = width, height = 960, 540

black = 0,0,0

pygame.camera.init()
device = pygame.camera.list_cameras()[0]
cam = pygame.camera.Camera(device, (1920, 1080))
cam.start()
img = cam.get_image()

screen = pygame.display.set_mode(size)
print(type(screen))
#exit()

bench = ibench.ImageBench(screen)
paths, dpoints = data.parse_datasets(keep_prob=.8, max_amount=100)
#datagen = ibench.create_data_generator(paths, points, 100, 0)()

model = model_v2('model_v2_4.h5')
points = []
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    #time.sleep(1.1)


    img = cam.get_image()
    
    img = ibench.resize(img, 150, 150)
    #img = ibench.to_array(picture)
    #dets = detector(img, 0)
    #print(dets)
    #screen.blit(img, img.get_rect())


    #print(pil.size)
    #pil = pil.resize((constants.width, constants.height), resample=Image.LANCZOS)


    #print(vals)
    
    #screen.blit(picture, (0,0))

    # img, points = ibench.permute('moe.png', [[123, 50]], 50)
    #img.show()
    #print(points)
    #bench.redraw(ibench.to_array('moe.png'), [[123, 123]])
    #d = datagen
    
    #img, points = ibench.create_dataset(paths, dpoints, 50, 0)
    #print('---', points)

    ppoints = model.predict(np.array([ibench.to_array(img)]), batch_size=1)[0]
    #print('+--', ibench.unflatten(ppoints))
    #print('++-', ppoints)

    

    #print(ibench.to_PIL(img).size)
    #points.append(ibench.unflatten(ppoints)[0])
    bench.redraw(img, ibench.unflatten(ppoints) + points, top_left=(0,0))
    #print(points)
    pygame.display.flip()
    screen.fill(black)

    #time.sleep(.5)
    

    


    # for j, d in enumerate(dets):
    #     #print(vals)

    #     center = d.center()
    #     width, height = d.width(), d.height()
    #     #print(center, width, height)

    #     pil = ibench.to_PIL(img)

    #     scale = .65

    #     left = center.x - width*scale
    #     top = center.y - height*scale
    #     right = center.x + width*scale
    #     bottom = center.y + height*scale

    #     pil = pil.crop((left, top, right, bottom))
    #     pil_temp = pil


    #     scale = constants.width / pil.size[0]

    #     predictable_image = pil.resize((constants.width, constants.height), resample=Image.LANCZOS)


    #     # arr = normalize(ibench.to_array(pil))
    #     # predictable = array([arr])
    #     predictable = array([ibench.to_array(predictable_image)])




    #     predictions = model.predict(predictable)[0]*(1/scale)

    #     bounding_box = [(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]
    #     points = predictions.tolist()
    #     points.append(bounding_box)
    #     #print(predictions.tolist())


    #     bench.redraw(pil, points, top_left=(left, top))
        #print(predictions.tolist())