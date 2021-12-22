import utility
import data
import model
import loss
from option import args
from checkpoint import Checkpoint
from trainer import Trainer
from multiprocessing.spawn import freeze_support
from GazeModule.gaze import GazeModel

utility.set_seed(args.seed)
checkpoint = Checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) 
    t = Trainer(args, loader, model, loss, checkpoint)

    def main():
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()

    if __name__ == '__main__':  # 중복 방지를 위한 사용
        freeze_support()  # 윈도우에서 파이썬이 자원을 효율적으로 사용하게 만들어준다.
        main()
