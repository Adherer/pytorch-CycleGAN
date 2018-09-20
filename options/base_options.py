import argparse            # 导入命令行解析的库文件
import os
from util import util
import torch
import models
import data

'''
train的命令：python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
test的命令：python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
'''
class BaseOptions():
    def __init__(self):
        self.initialized = False

    # 给parser初始化一些参数
    def initialize(self, parser):
        # required=True表示需要在命令行运行该python程序时显式给出
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='selects model to use for netD')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        parser.add_argument('--model', type=str, default='cycle_gan',
                            help='chooses which model to use. cycle_gan, pix2pix, test')
        parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # action - The basic type of action to be taken when this argument is encountered at the command line
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        self.initialized = True
        return parser


    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            '''
            设置一个解析器
            使用argparse的第一步就是创建一个解析器对象，并告诉它将会有些什么参数。那么当你的程序运行时，该解析器就可以用于处理命令行参数。
            解析器类是 ArgumentParser 。构造方法接收几个参数来设置用于程序帮助文本的描述信息以及其他全局的行为或设置。
            argparse.ArgumentDefaultsHelpFormatter 在每个选项的帮助信息后面输出他们对应的缺省值，如果有设置的话。
            之后再初始化解析器，并将add_argument后的解析器返回
            '''
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        # opt返回所有参数及其默认值(或者在命令行中给定的值)
        # 例如：Namespace(batch_size=1, dataroot='.', fineSize=256, input_nc=3, loadSize=286, ndf=64, netD='basic', ngf=64, output_nc=3)
        # 使用parser.parse_known_args()方法比parser.parse_args()方法鲁棒性更好，前者在运行时指定了程序中没有的参数时，不会报错，而是保存在第二个返回参数中
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model

        # model_option_setter得到静态modify_commandline_options方法入口
        model_option_setter = models.get_option_setter(model_name)

        # 传递base_options的option参数，并根据相应model的option参数，返回新的parser
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        # 同样的，根据dataset的option参数返回新的parser
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser

        return parser.parse_args()

    # 输出options，并将其存储到磁盘中
    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        # 将option参数写入opt.txt文件中
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    # 解析参数
    def parse(self):

        # 获取参数及其默认值
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        # 添加自定义后缀，training时一般不用，test_model中会设置自己的model_suffix
        # vars() 函数返回对象object的属性和属性值的字典对象。
        '''
        example：
        class Test():
            def __init__(self):
		        self.a = 'hello'

            obj = Test()
            print(vars(obj))     # {'a': 'hello'}
            print(obj.a.format(**vars(obj)))  # hello
            
        '''
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # 设置gpu id  1,2,3
        # 那么将会运行: torch.cuda.set_device(1) 哪里出问题了吗?
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

        # 这里是设置默认的gpu
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
