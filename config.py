import  argparse

args = argparse.ArgumentParser()
args.add_argument("--gpu", type=int, default=0,help="gpu")
args.add_argument('--Data-Vector-Length', type=int, default=50,
                  help='Setting data Dimensions')
args.add_argument('--centord-Vector-Length', type=int, default=30,
                  help='Setting centord Vector Dimensions')
args.add_argument('--Train-Ratio', type=float, default=0.8,
                  help='Training set ratio')
args.add_argument('--DATA-FILE', type=str, default='drift-50-25-4800',
                  help='Select data set')
args.add_argument('--DATA-SAMPLE-NUM', type=int, default=4800,
                  help='Number of data sets sampled')
args.add_argument('--RNN', type=str, default="RNN")
args.add_argument('--FAN', type=str, default="FAN")
args.add_argument('--FCN', type=str, default="FCN")
args.add_argument('--FQN', type=str, default="FQN")
args.add_argument('--FNN', type=str, default="FNN")
args.add_argument('--CNN', type=str, default="CNN")
args.add_argument('--PNN', type=str, default="PNN")
args.add_argument('--frame-size', type=int, default=200)
args.add_argument('--train-data', type=str, default="train")
args.add_argument('--test-data', type=str, default="test")

args.add_argument('--num-episode', type=int, default=600)
args.add_argument('--lr', type=float, default=0.01)
args.add_argument('--lr-scheduler-gamma', type=float, default=0.9)
args.add_argument('--lr_scheduler-step', type=int, default=30)
args.add_argument('--distillation-T', type=int, default=1)
args.add_argument('--distillation-type-alpha', type=int, default=0.25)
args.add_argument('--distillation-point-method', type=int, default=1)
args.add_argument('--student-lr', type=int, default=0.02)
args.add_argument('--student-num-episode', type=int, default=300)
args.add_argument('--student-frame-size', type=int, default=200)



args.add_argument('--Ns', type=int,
                  help='number of samples per class to use as support for training, default=5',
                  default=5)
args.add_argument('--Nc', type=int,
                  help='number of random classes per episode for training, default=4',
                  default=4)
args.add_argument('--Nq', type=int,
                  help='number of samples per class to use as query for training, default=5',
                  default=5)
args.add_argument('--iterations', type=int,
                  help='number of episodes per epoch, default=100',
                  default=200)

args.add_argument("--regularization", type=float, default=0.01,
                    help="regularization weight")
args = args.parse_args()
print(args)
