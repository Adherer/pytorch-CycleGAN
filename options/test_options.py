from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        # 根据BaseOptions添加初始化参数
        parser = BaseOptions.initialize(self, parser)
        # 添加一些额外的参数
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')

        # 与add_argument('--model', default='test')效果一致，会将BaseOptions中的--model参数覆盖
        parser.set_defaults(model='test')

        # 什么是loadSize? 什么是fineSize?
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        self.isTrain = False
        return parser
