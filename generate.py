import argparse, os
from net.generator import Generator
from net.discriminator import Discriminator
from net.learning import Learner
from util.image import generate_image


def generate(opt):
    
    generator = Generator()
    discriminator = Discriminator()
    
    learner = Learner(
        generator=generator,
        discriminator=discriminator
    )
    
    learner.restore()
    
    
    generate_image(generator = generator(),
                   input_image_path = opt.source,
                   target_image_path = opt.target,
                   only_predict = opt.predict_only,
                   save_dir = opt.save_dir
                   )
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type = str)
    parser.add_argument('--target', type = str)
    parser.add_argument('--predict-only', nargs='?', const=False, default=True)
    parser.add_argument('--save-dir', type=str, default="result/")
    opt = parser.parse_args()
    
    os.makedirs(opt.save_dir, exist_ok=True)
    
    # print(opt)
    
    generate(opt)
    
    