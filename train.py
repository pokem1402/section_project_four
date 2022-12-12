import argparse, os
from util.data import Dataset
from net.generator import Generator
from net.discriminator import Discriminator
from net.learning import Learner

TRAIN_PATH = os.path.join(os.getcwd(), 'Dataset', 'train')
TEST_PATH = os.path.join(os.getcwd(), 'Dataset', 'test')

def train(opt):
    ds = Dataset(train_path=TRAIN_PATH,
                 x_file_pattern=opt.x_pattern,
                 y_file_pattern=opt.y_pattern,
                 test_path=TEST_PATH,
                 buffer_size=opt.buffer_size,
                 batch_size=opt.batch_size,
                 seed=opt.seed)
    
    generator = Generator()
    discriminator = Discriminator()


    learner = Learner(
        generator=generator,
        discriminator=discriminator
    )
    
    if opt.resume:
        learner.restore()
    
    learner.fit(ds.train_ds, opt.epochs, ds.test_ds, save_only_last = opt.nosave)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--buffer-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--resume', nargs='?', const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", nargs='?', const=True, default=False, help="only save final checkpoint")
    parser.add_argument("--x-pattern", type=str, default="/sketch/*.jpg", help="the pattern of input image file names. ex) /sketch/*.jpg" )
    parser.add_argument("--y-pattern", type=str, default="/real/*.jpg", help="the pattern of target image files names. ex) /real/*.jpg")
    parser.add_argument("--seed", type=int, default=42)
    
    opt = parser.parse_args()
        
    train(opt)