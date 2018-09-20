import os
import torch
from collections import OrderedDict      # 有序字典,记录了每个键值对添加的顺序
from . import networks

# 作为所有model的基类
class BaseModel():

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        # 例如self.gpu_ids[0] = 0,则device的参数为torch.device('cuda:0')
        # 构造torch.device可以通过字符串/字符串和设备编号。 上述等同于: torch.device('cuda', 0)
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        # 存储路径，存储一些中间结果(阶段性的训练模型)以及可视化的web界面
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        # 在加载数据时是否缩放和裁剪图像
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True

        # 在相应的模型中会设置
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []

    # 在相应的模型中会设置(重写该函数)
    def set_input(self, input):
        self.input = input

    # 在相应的模型中会设置(重写该函数)
    def forward(self):
        pass

    # load and print networks; create schedulers
    # 加载和打印网络; 创建调度程序
    def setup(self, opt, parser=None):

        # 训练模式
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain or opt.continue_train:
            # which_epoch default='latest'
            self.load_networks(opt.which_epoch)
        # verbose action='store_true'
        # action = "store_true"，表示该选项不需要接收参数，直接设定args.verbose = True
        self.print_networks(opt.verbose)

    # make models eval mode during test time
    # 在测试时，将模型切换成eval mode
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                # getattr() 函数用于返回一个对象属性值。
                net = getattr(self, 'net' + name)
                net.eval()

    # used in test time, wrapping `forward` in no_grad()
    # so we don't save intermediate steps for backprop
    # 源码中没用上 暂时留着
    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    # 在相应的模型中重写该函数
    def optimize_parameters(self):
        pass

    # update learning rate (called once every epoch)
    # optimizers参数在相应的model中会添加
    # lr在base_options中有设置，default=0.0002
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()          # 有序字典,记录了每个键值对添加的顺序
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)    # getattr(self, name) 主要运用于相关模型在调用该方法时
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))   # 同get_current_visuals
        return errors_ret

    # save models to the disk
    # 将阶段性epoch的模型保存到磁盘上
    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # 回一个字典，保存着module的所有状态（state）
                    # net.module ==> 这时的module是torch.nn.DataParallel class的一个属性
                    torch.save(net.module.cpu().state_dict(), save_path)
                    # 当调用 .cuda() 的时候，submodule的参数也会转换为cuda Tensor
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # 私有方法
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)

                # torch.nn.DataParallel Multi-GPU layers 在模块级别上实现数据并行
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device

                # map_location表示数据要加载到哪，这里是加载到cuda:0上
                state_dict = torch.load(load_path, map_location=str(self.device))

                # del语句作用在变量上，而不是数据对象上。
                # 删除元数据
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()          # 返回一个tensor变量内所有元素个数，可以理解为矩阵内元素的个数
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=False to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        # 不是list则先转成list，方便遍历
        if not isinstance(nets, list):
            nets = [nets]

        # 返回一个 包含模型所有参数 的迭代器。
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
