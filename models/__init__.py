import importlib        # 动态导入模块
from models.base_model import BaseModel

'''
__init__.py是Python中package的标识，不能删除，一般来说在这个py文件中不会编写python模块
如果有编写模块，一般是一些很常用的模块，__init__.py中的模块在导入package时就导入到了相关文件中
'''

'''
这个函数主要是通过传入的model_name动态加载model.model_name_model模块
并且这里做了鲁棒性的测试，如果是BaseModel的子类才将model name返回
否则报错并退出程序
'''
def find_model_using_name(model_name):

    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    # issubclass() 方法用于判断参数 class 是否是类型参数 classinfo 的子类。
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

# 返回对应model_class的modify_commandline_options方法
def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options

# 根据opt.model的name返回指定的模型
def create_model(opt):
    model = find_model_using_name(opt.model)    # 返回相应model的class
    instance = model()                  # 创建model的对象实例
    instance.initialize(opt)            # 根据opt初始化instance实例
    print("model [%s] was created" % (instance.name()))
    return instance
